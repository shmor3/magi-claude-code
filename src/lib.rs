//! Claude Code ACP Agent Plugin
//!
//! WASM plugin that registers as an ACP agent with code generation,
//! code review, and code explanation capabilities. Communicates via
//! the ACP host functions (agent_register, agent_send, agent_receive).

use magi_pdk::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

const AGENT_NAME: &str = "claude-code";
const AGENT_DESCRIPTION: &str = "Code generation, review, and refactoring agent";

const CAPABILITIES: &[(&str, &str)] = &[
    (
        "code-generation",
        "Generate code from natural language descriptions",
    ),
    (
        "code-review",
        "Review code for bugs, style issues, and improvements",
    ),
    (
        "code-explanation",
        "Explain what a piece of code does",
    ),
    (
        "code-refactor",
        "Suggest refactoring improvements for code",
    ),
];

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentConfig {
    max_response_length: usize,
    auto_poll: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_response_length: 4096,
            auto_poll: true,
        }
    }
}

struct PluginState {
    agent_id: String,
    config: AgentConfig,
    messages_processed: u64,
    running: bool,
    started_at_ms: u64,
}

thread_local! {
    static STATE: RefCell<Option<PluginState>> = const { RefCell::new(None) };
}

fn with_state<T, F: FnOnce(&PluginState) -> T>(f: F) -> Option<T> {
    STATE.with(|s| s.borrow().as_ref().map(f))
}

fn with_state_mut<T, F: FnOnce(&mut PluginState) -> T>(f: F) -> Option<T> {
    STATE.with(|s| s.borrow_mut().as_mut().map(f))
}

fn config_from_datatype(dt: &DataType) -> AgentConfig {
    let mut config = AgentConfig::default();
    if let DataType::Map(map) = dt {
        if let Some(v) = map.get("max_response_length") {
            if let Some(n) = v.as_u32() {
                config.max_response_length = n as usize;
            }
        }
        if let Some(DataType::Bool(b)) = map.get("auto_poll") {
            config.auto_poll = *b;
        }
    }
    config
}

/// Handle a capability request from another agent.
fn handle_capability_request(capability: &str, payload: &serde_json::Value) -> serde_json::Value {
    let input = payload
        .get("input")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match capability {
        "code-generation" => {
            log_info(&format!("Code generation request: {} chars", input.len()));
            serde_json::json!({
                "status": "accepted",
                "capability": capability,
                "message": format!("Code generation request received ({} chars)", input.len()),
                "request_id": now_ms().to_string(),
            })
        }
        "code-review" => {
            log_info(&format!("Code review request: {} chars", input.len()));
            serde_json::json!({
                "status": "accepted",
                "capability": capability,
                "message": format!("Code review request received ({} chars)", input.len()),
                "request_id": now_ms().to_string(),
            })
        }
        "code-explanation" => {
            log_info(&format!("Code explanation request: {} chars", input.len()));
            serde_json::json!({
                "status": "accepted",
                "capability": capability,
                "message": format!("Code explanation request received ({} chars)", input.len()),
                "request_id": now_ms().to_string(),
            })
        }
        "code-refactor" => {
            log_info(&format!("Code refactor request: {} chars", input.len()));
            serde_json::json!({
                "status": "accepted",
                "capability": capability,
                "message": format!("Refactoring request received ({} chars)", input.len()),
                "request_id": now_ms().to_string(),
            })
        }
        other => {
            log_warn(&format!("Unknown capability requested: {}", other));
            serde_json::json!({
                "status": "error",
                "message": format!("Unknown capability: {}", other),
            })
        }
    }
}

/// Initialize the plugin — registers as an ACP agent.
#[plugin_fn]
pub fn init(input: Json<DataType>) -> FnResult<Json<DataType>> {
    let config = match &input.0 {
        DataType::Map(map) => map
            .get("config")
            .map(config_from_datatype)
            .unwrap_or_default(),
        _ => AgentConfig::default(),
    };

    log_info("Claude Code ACP agent initializing...");

    let agent_id = match agent_register(AGENT_NAME, AGENT_DESCRIPTION, CAPABILITIES) {
        Ok(id) => {
            log_info(&format!("Registered as ACP agent: {}", id));
            id
        }
        Err(e) => {
            log_error(&format!("Failed to register ACP agent: {}", e));
            return Ok(Json(
                DataType::map()
                    .insert("success", false)
                    .insert("error", format!("ACP registration failed: {}", e))
                    .build(),
            ));
        }
    };

    STATE.with(|s| {
        *s.borrow_mut() = Some(PluginState {
            agent_id: agent_id.clone(),
            config,
            messages_processed: 0,
            running: false,
            started_at_ms: 0,
        });
    });

    Ok(Json(
        DataType::map()
            .insert("success", true)
            .insert("agent_id", agent_id)
            .insert(
                "capabilities",
                DataType::Array(
                    CAPABILITIES
                        .iter()
                        .map(|(name, _)| DataType::String(name.to_string()))
                        .collect(),
                ),
            )
            .build(),
    ))
}

/// Start the plugin.
#[plugin_fn]
pub fn start(_: Json<DataType>) -> FnResult<Json<DataType>> {
    log_info("Claude Code ACP agent starting");
    let now = now_ms();
    with_state_mut(|state| {
        state.running = true;
        state.started_at_ms = now;
    });

    let _ = subscribe(&["agent.*"], None);

    Ok(Json(
        DataType::map()
            .insert("success", true)
            .insert("message", "Claude Code agent started")
            .build(),
    ))
}

/// Stop the plugin.
#[plugin_fn]
pub fn stop(_: Json<DataType>) -> FnResult<Json<DataType>> {
    log_info("Claude Code ACP agent stopping");
    with_state_mut(|state| {
        state.running = false;
    });

    Ok(Json(
        DataType::map()
            .insert("success", true)
            .insert("message", "Claude Code agent stopped")
            .build(),
    ))
}

/// Process incoming ACP messages.
#[plugin_fn]
pub fn process(_input: Json<DataType>) -> FnResult<Json<DataType>> {
    let running = with_state(|s| s.running).unwrap_or(false);
    if !running {
        return Ok(Json(
            DataType::map()
                .insert("success", false)
                .insert("error", "Agent not running")
                .build(),
        ));
    }

    // Poll for incoming ACP messages
    let messages = match agent_receive(10) {
        Ok(msgs) => msgs,
        Err(e) => {
            log_warn(&format!("Failed to receive messages: {}", e));
            return Ok(Json(
                DataType::map()
                    .insert("success", true)
                    .insert("messages_processed", DataType::Int32(0))
                    .build(),
            ));
        }
    };

    let mut processed = 0u64;
    let agent_id = with_state(|s| s.agent_id.clone()).unwrap_or_default();

    for msg in &messages {
        let from = msg.get("from").and_then(|v| v.as_str()).unwrap_or("");
        let payload = msg
            .get("payload_json")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .or_else(|| msg.get("payload").cloned())
            .unwrap_or(serde_json::Value::Null);

        let capability = payload
            .get("capability")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        log_info(&format!("Processing {} request from {}", capability, from));

        let response = handle_capability_request(capability, &payload);

        let reply_payload = serde_json::json!({
            "response": response,
            "agent_id": agent_id,
            "capability": capability,
        });

        if let Err(e) = agent_send(from, reply_payload) {
            log_error(&format!("Failed to send reply to {}: {}", from, e));
        }

        processed += 1;
    }

    with_state_mut(|state| {
        state.messages_processed += processed;
    });

    if processed > 0 {
        let _ = emit_event(serde_json::json!({
            "type": "PluginOutput",
            "data": {
                "plugin_id": agent_id,
                "messages_processed": processed,
            }
        }));
    }

    Ok(Json(
        DataType::map()
            .insert("success", true)
            .insert("messages_processed", DataType::Uint64(processed))
            .build(),
    ))
}

/// Get plugin status.
#[plugin_fn]
pub fn get_status(_: Json<DataType>) -> FnResult<Json<DataType>> {
    let (agent_id, count, running, started_at_ms) = with_state(|s| {
        (
            s.agent_id.clone(),
            s.messages_processed,
            s.running,
            s.started_at_ms,
        )
    })
    .unwrap_or((String::new(), 0, false, 0));

    let uptime_secs = if running && started_at_ms > 0 {
        ((now_ms() - started_at_ms) / 1000) as i32
    } else {
        0
    };

    Ok(Json(
        DataType::map()
            .insert("id", AGENT_NAME)
            .insert("name", "Claude Code Agent")
            .insert("agent_id", agent_id)
            .insert("running", running)
            .insert("uptime_secs", DataType::Int32(uptime_secs))
            .insert("messages_processed", DataType::Uint64(count))
            .insert(
                "capabilities",
                DataType::Array(
                    CAPABILITIES
                        .iter()
                        .map(|(name, _)| DataType::String(name.to_string()))
                        .collect(),
                ),
            )
            .build(),
    ))
}

/// Health check.
#[plugin_fn]
pub fn health_check(_: Json<DataType>) -> FnResult<Json<DataType>> {
    let running = with_state(|s| s.running).unwrap_or(false);
    Ok(Json(
        DataType::map()
            .insert("healthy", true)
            .insert("running", running)
            .insert("message", "Claude Code agent healthy")
            .build(),
    ))
}

/// Describe the plugin's capabilities.
#[plugin_fn]
pub fn describe(_: Json<DataType>) -> FnResult<Json<DataType>> {
    Ok(Json(
        DataType::map()
            .insert("name", AGENT_NAME)
            .insert("description", AGENT_DESCRIPTION)
            .insert("label", "acp")
            .insert(
                "capabilities",
                DataType::Array(
                    CAPABILITIES
                        .iter()
                        .map(|(name, desc)| {
                            DataType::map()
                                .insert("name", *name)
                                .insert("description", *desc)
                                .build()
                        })
                        .collect(),
                ),
            )
            .build(),
    ))
}

/// Config schema for the UI.
#[plugin_fn]
pub fn config_schema(_: Json<DataType>) -> FnResult<Json<DataType>> {
    Ok(Json(
        DataType::map()
            .insert(
                "max_response_length",
                DataType::map()
                    .insert("type", "number")
                    .insert("default", DataType::Int32(4096))
                    .insert("description", "Maximum response length in characters")
                    .build(),
            )
            .insert(
                "auto_poll",
                DataType::map()
                    .insert("type", "boolean")
                    .insert("default", true)
                    .insert("description", "Automatically poll for incoming messages")
                    .build(),
            )
            .build(),
    ))
}
