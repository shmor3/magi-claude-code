//! Text-LLM Plugin - Standalone text-to-text language model processing
//!
//! This WASM plugin processes text input and returns text output.
//! It follows the same pattern as instruct/reason plugins:
//! - Plugin is lightweight WASM (portable, sandboxed)
//! - Calls run_inference() which runs on the host with GPU/CUDA
//! - Host runs the Candle language model with GPU acceleration
//!
//! This is the correct architecture because:
//! - WASM cannot directly access CUDA/GPU
//! - Keeps plugins small and fast
//! - Host manages model loading and GPU resources

use magi_pdk::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

/// Plugin configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextLlmConfig {
    /// System prompt
    pub system_prompt: String,
    /// Model ID to use (e.g., "qwen2-0.5b", "deepseek-r1-1.5b")
    pub model_id: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling
    pub temperature: f32,
}

impl Default for TextLlmConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful AI assistant.".to_string(),
            model_id: "qwen2-0.5b".to_string(),
            max_tokens: 150,
            temperature: 0.7,
        }
    }
}

struct PluginState {
    config: TextLlmConfig,
    node_id: String,
    process_count: u64,
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

fn config_from_datatype(dt: &DataType) -> TextLlmConfig {
    let mut config = TextLlmConfig::default();
    if let DataType::Map(map) = dt {
        if let Some(DataType::String(s)) = map.get("system_prompt") {
            config.system_prompt = s.clone();
        }
        if let Some(DataType::String(s)) = map.get("model_id") {
            config.model_id = s.clone();
        }
        if let Some(v) = map.get("max_tokens") {
            if let Some(n) = v.as_u32() {
                config.max_tokens = n;
            } else if let Some(n) = v.as_i32() {
                config.max_tokens = n as u32;
            }
        }
        if let Some(DataType::Float32(f)) = map.get("temperature") {
            config.temperature = *f;
        }
        if let Some(DataType::Float64(f)) = map.get("temperature") {
            config.temperature = *f as f32;
        }
    }
    config
}

/// Run inference via host call_handler, with fallback for when host handler is unavailable
fn run_inference(prompt: &str, config: &TextLlmConfig) -> String {
    #[derive(Serialize)]
    struct InferenceRequest {
        prompt: String,
        model_id: String,
        max_tokens: u32,
        temperature: f32,
    }

    #[derive(Deserialize)]
    struct InferenceResponse {
        text: String,
    }

    let request = InferenceRequest {
        prompt: prompt.to_string(),
        model_id: config.model_id.clone(),
        max_tokens: config.max_tokens,
        temperature: config.temperature,
    };

    match call_handler::<_, InferenceResponse>("run_inference", &request) {
        Ok(response) => response.text,
        Err(e) => {
            log_warn(&format!(
                "Host inference unavailable ({}), using echo fallback",
                e
            ));
            format!("[model:{} not loaded] Echo: {}", config.model_id, prompt)
        }
    }
}

/// Initialize the plugin
#[plugin_fn]
pub fn init(input: Json<DataType>) -> FnResult<Json<DataType>> {
    let input_map = match &input.0 {
        DataType::Map(map) => map,
        _ => {
            return Ok(Json(DataType::Map(
                [
                    ("success".to_string(), DataType::Bool(false)),
                    (
                        "error".to_string(),
                        DataType::String("Input must be a map".to_string()),
                    ),
                ]
                .into_iter()
                .collect(),
            )))
        }
    };

    let node_id = input_map
        .get("node_id")
        .and_then(|v| v.as_str())
        .unwrap_or("text-llm")
        .to_string();

    // Parse config from input, falling back to defaults
    let config = input_map
        .get("config")
        .map(config_from_datatype)
        .unwrap_or_default();

    log_info(&format!(
        "Text-LLM plugin initializing with node: {}, model: {}, max_tokens: {}",
        node_id, config.model_id, config.max_tokens
    ));

    STATE.with(|s| {
        *s.borrow_mut() = Some(PluginState {
            config,
            node_id,
            process_count: 0,
            running: false,
            started_at_ms: 0,
        });
    });

    Ok(Json(DataType::Map(
        [
            ("success".to_string(), DataType::Bool(true)),
            ("error".to_string(), DataType::Null),
            (
                "capabilities".to_string(),
                DataType::Array(vec![
                    DataType::String("text_processing".to_string()),
                    DataType::String("language_model".to_string()),
                ]),
            ),
        ]
        .into_iter()
        .collect(),
    )))
}

/// Start the plugin
#[plugin_fn]
pub fn start(_: Json<DataType>) -> FnResult<Json<DataType>> {
    log_info("Text-LLM plugin starting");

    let now = now_ms();
    with_state_mut(|state| {
        state.running = true;
        state.started_at_ms = now;
    });

    Ok(Json(DataType::Map(
        [
            ("success".to_string(), DataType::Bool(true)),
            (
                "message".to_string(),
                DataType::String("Text-LLM plugin started".to_string()),
            ),
        ]
        .into_iter()
        .collect(),
    )))
}

/// Stop the plugin
#[plugin_fn]
pub fn stop(_: Json<DataType>) -> FnResult<Json<DataType>> {
    log_info("Text-LLM plugin stopping");

    with_state_mut(|state| {
        state.running = false;
    });

    Ok(Json(DataType::Map(
        [
            ("success".to_string(), DataType::Bool(true)),
            (
                "message".to_string(),
                DataType::String("Text-LLM plugin stopped".to_string()),
            ),
        ]
        .into_iter()
        .collect(),
    )))
}

#[plugin_fn]
pub fn process_text(input: Json<DataType>) -> FnResult<Json<DataType>> {
    let input_map = match &input.0 {
        DataType::Map(map) => map,
        _ => {
            return Ok(Json(DataType::Map(
                [
                    ("success".to_string(), DataType::Bool(false)),
                    (
                        "error".to_string(),
                        DataType::String("Input must be a map".to_string()),
                    ),
                ]
                .into_iter()
                .collect(),
            )));
        }
    };
    let text = input_map
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let config = with_state(|s| s.config.clone()).unwrap_or_default();

    log_info(&format!("Processing text input: {} chars", text.len()));

    with_state_mut(|state| {
        state.process_count += 1;
    });

    // Build prompt and run inference via host
    let prompt = format!("{}\n\nUser: {}\n\nAssistant:", config.system_prompt, text);
    let generated_text = run_inference(&prompt, &config);

    // Publish output for graph routing
    let node_id = with_state(|s| s.node_id.clone()).unwrap_or_else(|| "text-llm".to_string());
    let _ = publish_event(
        "plugin.output",
        &serde_json::json!({
            "source": node_id,
            "port": "output",
            "data_type": "text",
            "value": generated_text,
        }),
    );

    Ok(Json(DataType::Map(
        [
            ("success".to_string(), DataType::Bool(true)),
            (
                "output".to_string(),
                DataType::String(generated_text),
            ),
        ]
        .into_iter()
        .collect(),
    )))
}

/// Get plugin status
#[plugin_fn]
pub fn get_status(_: Json<DataType>) -> FnResult<Json<DataType>> {
    let (count, running, started_at_ms, config) = with_state(|s| {
        (
            s.process_count,
            s.running,
            s.started_at_ms,
            s.config.clone(),
        )
    })
    .unwrap_or((0, false, 0, TextLlmConfig::default()));

    let uptime_secs = if running && started_at_ms > 0 {
        ((now_ms() - started_at_ms) / 1000) as i32
    } else {
        0
    };

    Ok(Json(DataType::Map(
        [
            ("id".to_string(), DataType::String("text-llm".to_string())),
            (
                "name".to_string(),
                DataType::String("Text LLM Plugin".to_string()),
            ),
            ("running".to_string(), DataType::Bool(running)),
            ("uptime_secs".to_string(), DataType::Int32(uptime_secs)),
            (
                "frames_processed".to_string(),
                DataType::Int64(count as i64),
            ),
            ("fps".to_string(), DataType::Float32(0.0)),
            ("last_error".to_string(), DataType::Null),
            (
                "metrics".to_string(),
                DataType::Map(
                    [
                        ("process_count".to_string(), DataType::Int64(count as i64)),
                        (
                            "model".to_string(),
                            DataType::String(config.model_id),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                ),
            ),
        ]
        .into_iter()
        .collect(),
    )))
}

/// Handle input from graph data injection (required for graph execution)
#[plugin_fn]
pub fn handle_input(input: Json<DataType>) -> FnResult<Json<DataType>> {
    let input_map = match &input.0 {
        DataType::Map(map) => map,
        _ => {
            return Ok(Json(DataType::Map(
                [
                    ("success".to_string(), DataType::Bool(false)),
                    (
                        "error".to_string(),
                        DataType::String("Input must be a map".to_string()),
                    ),
                ]
                .into_iter()
                .collect(),
            )))
        }
    };

    let value = input_map
        .get("value")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let target_node_id = input_map
        .get("target_node_id")
        .and_then(|v| v.as_str())
        .unwrap_or("text-llm")
        .to_string();

    if value.is_empty() {
        log_info("Empty input received, skipping processing");
        return Ok(Json(DataType::Map(
            [
                ("success".to_string(), DataType::Bool(true)),
                (
                    "message".to_string(),
                    DataType::String("Empty input, no processing needed".to_string()),
                ),
            ]
            .into_iter()
            .collect(),
        )));
    }

    log_info(&format!("Processing text input: {}", value));

    let config = with_state(|s| s.config.clone()).unwrap_or_default();

    with_state_mut(|state| {
        state.process_count += 1;
    });

    let prompt = format!("{}\n\nUser: {}\n\nAssistant:", config.system_prompt, value);
    let generated_text = run_inference(&prompt, &config);

    let _ = publish_event(
        "plugin.output",
        &serde_json::json!({
            "source": target_node_id,
            "port": "output",
            "data_type": "text",
            "value": generated_text,
        }),
    );

    Ok(Json(DataType::Map(
        [
            ("success".to_string(), DataType::Bool(true)),
            (
                "output".to_string(),
                DataType::String(generated_text.clone()),
            ),
            (
                "message".to_string(),
                DataType::String("Text processed successfully".to_string()),
            ),
        ]
        .into_iter()
        .collect(),
    )))
}

/// Health check
#[plugin_fn]
pub fn health_check(_: Json<DataType>) -> FnResult<Json<DataType>> {
    let running = with_state(|s| s.running).unwrap_or(false);
    Ok(Json(DataType::Map(
        [
            ("healthy".to_string(), DataType::Bool(true)),
            ("running".to_string(), DataType::Bool(running)),
            (
                "message".to_string(),
                DataType::String("Text-LLM plugin healthy".to_string()),
            ),
        ]
        .into_iter()
        .collect(),
    )))
}
