//! MCP (Model Context Protocol) client hub for pulsar-serve.
//!
//! Isolates the async rmcp SDK + a private tokio runtime behind a synchronous
//! API (block_on bridge) so the hand-rolled sync HTTP server stays sync
//! everywhere except this one leaf. Enabled only when pulsar-serve is launched
//! with --webui-mcp-proxy.
//!
//! ponytail: a single process-global hub. pulsar-serve is sequential
//! single-user localhost, so no per-request isolation; ceiling is per-session
//! hubs if multi-tenant ever matters.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rmcp::model::{CallToolRequestParams, ClientInfo, InitializeRequestParams, Tool};
use rmcp::service::{RunningService, ServiceExt};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransport;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::RoleClient;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::process::Command;

/// One MCP server entry in mcp.json. Untagged so the JSON form matches the
/// Claude-Code convention: command+args => stdio, url => remote streamable-http.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpServerCfg {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        #[serde(default)]
        allow: Vec<String>,
        #[serde(default)]
        deny: Vec<String>,
        #[serde(default)]
        disabled: bool,
        #[serde(default)]
        timeout_s: Option<u64>,
    },
    Remote {
        url: String,
        /// "http" (streamable). sse folded into streamable-http since rmcp 0.11.
        #[serde(default)]
        transport: Option<String>,
        #[serde(default)]
        headers: BTreeMap<String, String>,
        #[serde(default)]
        allow: Vec<String>,
        #[serde(default)]
        deny: Vec<String>,
        #[serde(default)]
        disabled: bool,
        #[serde(default)]
        timeout_s: Option<u64>,
    },
}

impl McpServerCfg {
    fn disabled(&self) -> bool {
        match self {
            Self::Stdio { disabled, .. } | Self::Remote { disabled, .. } => *disabled,
        }
    }
    fn allow(&self) -> &[String] {
        match self {
            Self::Stdio { allow, .. } | Self::Remote { allow, .. } => allow,
        }
    }
    fn deny(&self) -> &[String] {
        match self {
            Self::Stdio { deny, .. } | Self::Remote { deny, .. } => deny,
        }
    }
    fn timeout(&self) -> Duration {
        let s = match self {
            Self::Stdio { timeout_s, .. } | Self::Remote { timeout_s, .. } => *timeout_s,
        };
        Duration::from_secs(s.unwrap_or(30))
    }
    fn transport_kind(&self) -> &'static str {
        match self {
            Self::Stdio { .. } => "stdio",
            Self::Remote { .. } => "http",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: BTreeMap<String, McpServerCfg>,
}

// RunningService is not Clone, so the conn holds an Arc and dispatch clones
// the Arc into the async block. The second type param is the self-identity
// payload returned by ClientInfo::default().serve(...).
type ClientService = RunningService<RoleClient, InitializeRequestParams>;

struct Conn {
    name: String,
    cfg: McpServerCfg,
    client: Option<Arc<ClientService>>,
    tools: Vec<Tool>,
    error: Option<String>,
}

/// Synchronous facade over the async rmcp clients. Every public method is
/// safe to call from the sync HTTP thread (internally block_on's on a private
/// multi-thread runtime).
pub struct McpHub {
    rt: tokio::runtime::Runtime,
    conns: Mutex<Vec<Conn>>,
    /// namespaced tool names ("server__tool") disabled at runtime via the webui.
    overrides: Mutex<HashSet<String>>,
    config_path: Option<PathBuf>,
}

impl McpHub {
    pub fn new(config_path: Option<&std::path::Path>) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(2)
            .build()
            .expect("mcp tokio runtime");
        let config_path = config_path.map(PathBuf::from);
        let mut conns: Vec<Conn> = Vec::new();
        if let Some(p) = &config_path {
            if let Ok(txt) = std::fs::read_to_string(p) {
                if let Ok(cfg) = serde_json::from_str::<McpConfig>(&txt) {
                    for (name, c) in cfg.mcp_servers {
                        conns.push(Conn {
                            name,
                            cfg: c,
                            client: None,
                            tools: Vec::new(),
                            error: None,
                        });
                    }
                }
            }
        }
        McpHub {
            rt,
            conns: Mutex::new(conns),
            overrides: Mutex::new(HashSet::new()),
            config_path,
        }
    }

    /// Connect every configured server (best-effort; failures land in Conn::error).
    pub fn connect_all(&self) {
        let names: Vec<String> = self.conns.lock().unwrap().iter().map(|c| c.name.clone()).collect();
        for name in names {
            self.connect_one(&name);
        }
    }

    fn connect_one(&self, name: &str) {
        // Drop any old client before reconnecting (cancel on the runtime).
        let cfg = {
            let mut conns = self.conns.lock().unwrap();
            let Some(c) = conns.iter_mut().find(|c| c.name == name) else {
                return;
            };
            c.client = None;
            c.tools.clear();
            c.error = None;
            c.cfg.clone()
        };
        if cfg.disabled() {
            let mut conns = self.conns.lock().unwrap();
            if let Some(c) = conns.iter_mut().find(|c| c.name == name) {
                c.error = Some("disabled".into());
            }
            return;
        }
        let res = self.rt.block_on(async {
            match &cfg {
                McpServerCfg::Stdio { command, args, env, .. } => {
                    // ConfigureCommandExt::configure takes self by value and
                    // returns the configured Self; chain it so cmd stays owned.
                    let cmd = Command::new(command).configure(|c| {
                        for a in args {
                            c.arg(a);
                        }
                        for (k, v) in env {
                            c.env(k, v);
                        }
                    });
                    let transport = TokioChildProcess::new(cmd).map_err(|e| e.to_string())?;
                    ClientInfo::default()
                        .serve(transport)
                        .await
                        .map_err(|e| e.to_string())
                }
                McpServerCfg::Remote { url, .. } => {
                    // rmcp 3.0.1: StreamableHttpClientTransport exposes
                    // from_uri / from_config constructors, not ::new.
                    let transport = StreamableHttpClientTransport::from_uri(url.clone());
                    ClientInfo::default()
                        .serve(transport)
                        .await
                        .map_err(|e| e.to_string())
                }
            }
        });
        match res {
            Ok(client) => {
                let client = Arc::new(client);
                let tools = self.rt.block_on(async { client.list_all_tools().await });
                match tools {
                    Ok(t) => {
                        let mut conns = self.conns.lock().unwrap();
                        if let Some(c) = conns.iter_mut().find(|c| c.name == name) {
                            c.client = Some(client);
                            c.tools = t;
                        }
                    }
                    Err(e) => self.set_err(name, format!("list_tools: {e}")),
                }
            }
            Err(e) => self.set_err(name, e),
        }
    }

    fn set_err(&self, name: &str, e: String) {
        let mut conns = self.conns.lock().unwrap();
        if let Some(c) = conns.iter_mut().find(|c| c.name == name) {
            c.error = Some(e);
        }
    }

    pub fn has_enabled_tools(&self) -> bool {
        !self.enabled_tools_as_openai().is_empty()
    }

    /// All enabled tools across servers as OpenAI function specs, namespaced
    /// `server__tool` to prevent collisions.
    pub fn enabled_tools_as_openai(&self) -> Vec<Value> {
        let conns = self.conns.lock().unwrap();
        let overrides = self.overrides.lock().unwrap();
        let mut out = Vec::new();
        for c in conns.iter() {
            if c.client.is_none() {
                continue;
            }
            for t in &c.tools {
                let ns = format!("{}__{}", c.name, t.name);
                if overrides.contains(&ns) {
                    continue;
                }
                if !permitted(&c.cfg, &t.name) {
                    continue;
                }
                out.push(json!({
                    "type": "function",
                    "function": {
                        "name": ns,
                        "description": t.description.clone().unwrap_or_default(),
                        "parameters": schema_value(&t),
                    }
                }));
            }
        }
        out
    }

    /// Run a tool by its namespaced name. Returns the textual/JSON result the
    /// model sees inside `<tool_result>`. Never panics.
    pub fn dispatch_sync(&self, namespaced: &str, args: &str) -> String {
        let Some((server, tool)) = namespaced.split_once("__") else {
            return format!("error: malformed tool name {namespaced}");
        };
        let (client, timeout) = {
            let conns = self.conns.lock().unwrap();
            let Some(c) = conns.iter().find(|c| c.name == server) else {
                return format!("error: unknown server {server}");
            };
            if !permitted(&c.cfg, tool) {
                return format!("error: {namespaced} not permitted by allow/deny");
            }
            let Some(client) = c.client.clone() else {
                return format!("error: {} not connected", c.name);
            };
            (client, c.cfg.timeout())
        };
        let arg_map = parse_args(args);
        let params = {
            let mut p = CallToolRequestParams::new(tool.to_string());
            p.arguments = Some(arg_map);
            p
        };
        let result = self.rt.block_on(async move {
            tokio::time::timeout(timeout, client.call_tool(params)).await
        });
        match result {
            Ok(Ok(r)) => render_result(&r),
            Ok(Err(e)) => format!("error: {namespaced} call failed: {e}"),
            Err(_) => format!("error: {namespaced} timed out after {}s", timeout.as_secs()),
        }
    }

    /// Snapshot for GET /mcp/status.
    pub fn status_json(&self) -> Value {
        let conns = self.conns.lock().unwrap();
        let overrides = self.overrides.lock().unwrap();
        let servers = conns
            .iter()
            .map(|c| {
                let tools: Vec<Value> = c
                    .tools
                    .iter()
                    .map(|t| {
                        let ns = format!("{}__{}", c.name, t.name);
                        json!({
                            "name": t.name,
                            "namespaced": ns,
                            "description": t.description.clone().unwrap_or_default(),
                            "enabled": !overrides.contains(&ns) && permitted(&c.cfg, &t.name),
                        })
                    })
                    .collect();
                json!({
                    "name": c.name,
                    "transport": c.cfg.transport_kind(),
                    "connected": c.client.is_some(),
                    "disabled": c.cfg.disabled(),
                    "error": c.error.clone(),
                    "tools": tools,
                    // raw cfg so the webui edit form can repopulate every field
                    "config": serde_json::to_value(&c.cfg).unwrap_or(Value::Null),
                })
            })
            .collect::<Vec<_>>();
        json!({ "servers": servers })
    }

    /// Enable/disable a namespaced tool at runtime (webui toggle).
    pub fn toggle(&self, tool: &str, disabled: bool) {
        let mut ov = self.overrides.lock().unwrap();
        if disabled {
            ov.insert(tool.to_string());
        } else {
            ov.remove(tool);
        }
    }

    /// Add or replace a server, reconnect, persist.
    pub fn upsert_server(&self, name: &str, cfg: McpServerCfg) {
        {
            let mut conns = self.conns.lock().unwrap();
            if let Some(c) = conns.iter_mut().find(|c| c.name == name) {
                c.cfg = cfg.clone();
                c.client = None;
                c.tools.clear();
                c.error = None;
            } else {
                conns.push(Conn {
                    name: name.into(),
                    cfg: cfg.clone(),
                    client: None,
                    tools: Vec::new(),
                    error: None,
                });
            }
        }
        self.save_config();
        self.connect_one(name);
    }

    /// Remove a server, persist. Dropping the Arc<ClientService> (when the last
    /// in-flight dispatch finishes) cancels the background task via Drop.
    pub fn remove_server(&self, name: &str) {
        {
            let mut conns = self.conns.lock().unwrap();
            if let Some(pos) = conns.iter().position(|c| c.name == name) {
                conns.remove(pos);
            }
        }
        let prefix = format!("{name}__");
        let mut ov = self.overrides.lock().unwrap();
        ov.retain(|t| !t.starts_with(&prefix));
        drop(ov);
        self.save_config();
    }

    fn save_config(&self) {
        let Some(p) = &self.config_path else { return };
        let conns = self.conns.lock().unwrap();
        let mut m = BTreeMap::new();
        for c in conns.iter() {
            m.insert(c.name.clone(), c.cfg.clone());
        }
        let cfg = McpConfig { mcp_servers: m };
        let Ok(txt) = serde_json::to_string_pretty(&cfg) else {
            return;
        };
        let tmp = p.with_extension("json.tmp");
        if std::fs::write(&tmp, txt).is_ok() {
            let _ = std::fs::rename(&tmp, p);
        }
    }
}

fn permitted(cfg: &McpServerCfg, tool: &str) -> bool {
    if cfg.deny().iter().any(|d| d == tool) {
        return false;
    }
    let allow = cfg.allow();
    if !allow.is_empty() && !allow.iter().any(|a| a == tool) {
        return false;
    }
    true
}

fn parse_args(args: &str) -> Map<String, Value> {
    let v: Value = serde_json::from_str(args).unwrap_or(Value::Null);
    match v {
        Value::Object(m) => m,
        other => {
            let mut m = Map::new();
            m.insert("value".into(), other);
            m
        }
    }
}

fn schema_value(t: &Tool) -> Value {
    // rmcp 3.0.1: Tool::input_schema is Arc<Map<String, Value>> (not Optional);
    // deref + clone yields the owned Map Value::Object expects.
    Value::Object(t.input_schema.as_ref().clone())
}

fn render_result(r: &rmcp::model::CallToolResult) -> String {
    let mut out = String::new();
    if r.is_error.unwrap_or(false) {
        out.push_str("[tool error] ");
    }
    for block in &r.content {
        // ContentBlock variant names churn across rmcp versions; go through
        // serde so this never fails to compile on a minor bump.
        let v = serde_json::to_value(block).unwrap_or(Value::Null);
        if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
            out.push_str(t);
        } else {
            out.push_str(&v.to_string());
        }
        out.push('\n');
    }
    out.trim_end().to_string()
}

/// Build the set of namespaced tool names (handy for callers that only need
/// the dispatch surface); currently unused but kept for the webui status probe.
#[allow(dead_code)]
fn _namespaced_set(s: &McpHub) -> BTreeSet<String> {
    s.enabled_tools_as_openai()
        .iter()
        .filter_map(|v| v["function"]["name"].as_str().map(String::from))
        .collect()
}
