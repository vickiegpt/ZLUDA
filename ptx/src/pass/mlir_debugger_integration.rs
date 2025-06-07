// mlir_debugger_integration.rs - Advanced MLIR Debugger Integration
// This module provides comprehensive debugging tools for MLIR dialects including
// GDB/LLDB integration, breakpoint management, and runtime inspection.

use super::mlir_debug_framework::*;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

/// Advanced MLIR debugger with multi-backend support
pub struct MlirDebugger {
    debug_context: Arc<Mutex<UniversalMlirDebugContext>>,
    active_breakpoints: HashMap<u32, DebugBreakpoint>,
    execution_state: ExecutionState,
    backend: DebugBackend,
    symbol_table: SymbolTable,
    stack_frames: Vec<StackFrame>,
    watch_expressions: HashMap<u32, WatchExpression>,
    debug_session: Option<DebugSession>,
}

/// Debug backend selection
#[derive(Debug, Clone)]
pub enum DebugBackend {
    GDB {
        executable_path: String,
        args: Vec<String>,
    },
    LLDB {
        executable_path: String,
        args: Vec<String>,
    },
    MlirDebugger {
        mlir_opt_path: String,
        debug_options: Vec<String>,
    },
    Custom {
        name: String,
        command: String,
        args: Vec<String>,
    },
}

/// Execution state tracking
#[derive(Debug, Clone)]
pub enum ExecutionState {
    NotStarted,
    Running,
    Paused {
        location: UniversalDebugLocation,
        reason: PauseReason,
    },
    Finished {
        exit_code: i32,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Clone)]
pub enum PauseReason {
    Breakpoint(u32),
    StepComplete,
    Exception(String),
    UserInterrupt,
}

/// Enhanced breakpoint with conditions and actions
#[derive(Debug, Clone)]
pub struct DebugBreakpoint {
    pub id: u32,
    pub location: UniversalDebugLocation,
    pub condition: Option<String>,
    pub hit_count: u32,
    pub enabled: bool,
    pub actions: Vec<BreakpointAction>,
    pub breakpoint_type: BreakpointType,
}

#[derive(Debug, Clone)]
pub enum BreakpointType {
    Line,
    Function,
    Instruction,
    Memory {
        address: u64,
        size: usize,
        access_type: MemoryAccessType,
    },
    Exception(String),
}

#[derive(Debug, Clone)]
pub enum MemoryAccessType {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub enum BreakpointAction {
    Print(String),
    Log(String),
    Execute(String),
    Continue,
}

/// Symbol table for variable and function lookup
#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub functions: HashMap<String, FunctionSymbol>,
    pub variables: HashMap<String, VariableSymbol>,
    pub types: HashMap<String, TypeSymbol>,
    pub modules: HashMap<String, ModuleSymbol>,
}

#[derive(Debug, Clone)]
pub struct FunctionSymbol {
    pub name: String,
    pub mangled_name: Option<String>,
    pub location: UniversalDebugLocation,
    pub parameters: Vec<ParameterSymbol>,
    pub return_type: String,
    pub inline_info: Option<InlineInfo>,
}

#[derive(Debug, Clone)]
pub struct ParameterSymbol {
    pub name: String,
    pub type_name: String,
    pub location: Option<UniversalDebugLocation>,
}

#[derive(Debug, Clone)]
pub struct VariableSymbol {
    pub name: String,
    pub type_name: String,
    pub location: UniversalDebugLocation,
    pub scope: String,
    pub storage_location: StorageLocation,
}

#[derive(Debug, Clone)]
pub enum StorageLocation {
    Register(String),
    Memory(u64),
    Stack(i64),
    Constant(String),
}

#[derive(Debug, Clone)]
pub struct TypeSymbol {
    pub name: String,
    pub size_bytes: u64,
    pub alignment: u32,
    pub members: Vec<TypeMember>,
}

#[derive(Debug, Clone)]
pub struct TypeMember {
    pub name: String,
    pub type_name: String,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct ModuleSymbol {
    pub name: String,
    pub dialect: MlirDialect,
    pub file_path: String,
}

#[derive(Debug, Clone)]
pub struct InlineInfo {
    pub inlined_at: UniversalDebugLocation,
    pub call_site: UniversalDebugLocation,
}

/// Stack frame representation
#[derive(Debug, Clone)]
pub struct StackFrame {
    pub function_name: String,
    pub location: UniversalDebugLocation,
    pub local_variables: HashMap<String, VariableValue>,
    pub frame_id: u32,
}

#[derive(Debug, Clone)]
pub struct VariableValue {
    pub type_name: String,
    pub value: String,
    pub is_optimized_out: bool,
    pub children: Option<Vec<VariableValue>>,
}

/// Watch expression for monitoring values
#[derive(Debug, Clone)]
pub struct WatchExpression {
    pub id: u32,
    pub expression: String,
    pub enabled: bool,
    pub last_value: Option<String>,
    pub change_count: u32,
}

/// Debug session management
#[derive(Debug)]
pub struct DebugSession {
    pub session_id: String,
    pub start_time: std::time::SystemTime,
    pub target_executable: String,
    pub debug_log: Vec<DebugLogEntry>,
}

#[derive(Debug, Clone)]
pub struct DebugLogEntry {
    pub timestamp: std::time::SystemTime,
    pub level: LogLevel,
    pub message: String,
    pub location: Option<UniversalDebugLocation>,
}

#[derive(Debug, Clone)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
}

impl MlirDebugger {
    pub fn new(debug_context: UniversalMlirDebugContext, backend: DebugBackend) -> Self {
        Self {
            debug_context: Arc::new(Mutex::new(debug_context)),
            active_breakpoints: HashMap::new(),
            execution_state: ExecutionState::NotStarted,
            backend,
            symbol_table: SymbolTable::new(),
            stack_frames: Vec::new(),
            watch_expressions: HashMap::new(),
            debug_session: None,
        }
    }

    /// Start a debugging session
    pub fn start_session(&mut self, target_executable: &str) -> Result<String, String> {
        let session_id = format!(
            "mlir_debug_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        self.debug_session = Some(DebugSession {
            session_id: session_id.clone(),
            start_time: std::time::SystemTime::now(),
            target_executable: target_executable.to_string(),
            debug_log: Vec::new(),
        });

        self.log_debug_event(
            LogLevel::Info,
            &format!("Started debug session: {}", session_id),
            None,
        );

        // Initialize symbol table
        self.build_symbol_table(target_executable)?;

        Ok(session_id)
    }

    /// Set a breakpoint with advanced options
    pub fn set_breakpoint_advanced(
        &mut self,
        location: UniversalDebugLocation,
        condition: Option<String>,
        actions: Vec<BreakpointAction>,
        breakpoint_type: BreakpointType,
    ) -> Result<u32, String> {
        let breakpoint_id = self.active_breakpoints.len() as u32 + 1;

        let breakpoint = DebugBreakpoint {
            id: breakpoint_id,
            location: location.clone(),
            condition,
            hit_count: 0,
            enabled: true,
            actions,
            breakpoint_type,
        };

        // Set breakpoint in backend debugger
        match &self.backend {
            DebugBackend::GDB { .. } => {
                self.set_gdb_breakpoint(&breakpoint)?;
            }
            DebugBackend::LLDB { .. } => {
                self.set_lldb_breakpoint(&breakpoint)?;
            }
            DebugBackend::MlirDebugger { .. } => {
                self.set_mlir_breakpoint(&breakpoint)?;
            }
            DebugBackend::Custom { .. } => {
                self.set_custom_breakpoint(&breakpoint)?;
            }
        }

        self.active_breakpoints.insert(breakpoint_id, breakpoint);

        self.log_debug_event(
            LogLevel::Info,
            &format!(
                "Set breakpoint {} at {}:{}:{}",
                breakpoint_id, location.file, location.line, location.column
            ),
            Some(location),
        );

        Ok(breakpoint_id)
    }

    /// Add a watch expression
    pub fn add_watch_expression(&mut self, expression: &str) -> Result<u32, String> {
        let watch_id = self.watch_expressions.len() as u32 + 1;

        let watch = WatchExpression {
            id: watch_id,
            expression: expression.to_string(),
            enabled: true,
            last_value: None,
            change_count: 0,
        };

        self.watch_expressions.insert(watch_id, watch);

        self.log_debug_event(
            LogLevel::Info,
            &format!("Added watch expression {}: {}", watch_id, expression),
            None,
        );

        Ok(watch_id)
    }

    /// Step through execution
    pub fn step_instruction(&mut self) -> Result<ExecutionState, String> {
        match &self.backend {
            DebugBackend::GDB { .. } => {
                self.step_gdb()?;
            }
            DebugBackend::LLDB { .. } => {
                self.step_lldb()?;
            }
            DebugBackend::MlirDebugger { .. } => {
                self.step_mlir_debugger()?;
            }
            DebugBackend::Custom { .. } => {
                self.step_custom()?;
            }
        }

        // Update execution state
        self.update_execution_state()?;

        // Evaluate watch expressions
        self.evaluate_watch_expressions()?;

        Ok(self.execution_state.clone())
    }

    /// Get variable value at current location
    pub fn get_variable_value(&self, variable_name: &str) -> Result<VariableValue, String> {
        // Look up variable in symbol table
        if let Some(symbol) = self.symbol_table.variables.get(variable_name) {
            match &self.backend {
                DebugBackend::GDB { .. } => self.get_gdb_variable_value(variable_name, symbol),
                DebugBackend::LLDB { .. } => self.get_lldb_variable_value(variable_name, symbol),
                DebugBackend::MlirDebugger { .. } => {
                    self.get_mlir_variable_value(variable_name, symbol)
                }
                DebugBackend::Custom { .. } => {
                    self.get_custom_variable_value(variable_name, symbol)
                }
            }
        } else {
            Err(format!(
                "Variable '{}' not found in symbol table",
                variable_name
            ))
        }
    }

    /// Generate comprehensive debugging report
    pub fn generate_debug_report(&self) -> String {
        let mut report = String::new();

        report.push_str("===== MLIR DEBUGGING REPORT =====\n");

        if let Some(session) = &self.debug_session {
            report.push_str(&format!("Session ID: {}\n", session.session_id));
            report.push_str(&format!("Target: {}\n", session.target_executable));
            report.push_str(&format!("Start Time: {:?}\n", session.start_time));
        }

        report.push_str(&format!("Backend: {:?}\n", self.backend));
        report.push_str(&format!("Execution State: {:?}\n", self.execution_state));
        report.push_str("\n");

        // Breakpoints
        report.push_str("=== BREAKPOINTS ===\n");
        for (id, bp) in &self.active_breakpoints {
            report.push_str(&format!(
                "  [{}] {}:{}:{} (hits: {}, enabled: {})\n",
                id,
                bp.location.file,
                bp.location.line,
                bp.location.column,
                bp.hit_count,
                bp.enabled
            ));
            if let Some(ref condition) = bp.condition {
                report.push_str(&format!("      Condition: {}\n", condition));
            }
        }
        report.push_str("\n");

        // Watch expressions
        report.push_str("=== WATCH EXPRESSIONS ===\n");
        for (id, watch) in &self.watch_expressions {
            report.push_str(&format!(
                "  [{}] {} (changes: {}, enabled: {})\n",
                id, watch.expression, watch.change_count, watch.enabled
            ));
            if let Some(ref value) = watch.last_value {
                report.push_str(&format!("      Last Value: {}\n", value));
            }
        }
        report.push_str("\n");

        // Stack trace
        report.push_str("=== STACK TRACE ===\n");
        for (i, frame) in self.stack_frames.iter().enumerate() {
            report.push_str(&format!(
                "  #{} {} at {}:{}:{}\n",
                i,
                frame.function_name,
                frame.location.file,
                frame.location.line,
                frame.location.column
            ));
            for (var_name, var_value) in &frame.local_variables {
                report.push_str(&format!(
                    "      {}: {} = {}\n",
                    var_name, var_value.type_name, var_value.value
                ));
            }
        }
        report.push_str("\n");

        // Symbol table summary
        report.push_str("=== SYMBOL TABLE SUMMARY ===\n");
        report.push_str(&format!(
            "  Functions: {}\n",
            self.symbol_table.functions.len()
        ));
        report.push_str(&format!(
            "  Variables: {}\n",
            self.symbol_table.variables.len()
        ));
        report.push_str(&format!("  Types: {}\n", self.symbol_table.types.len()));
        report.push_str(&format!("  Modules: {}\n", self.symbol_table.modules.len()));
        report.push_str("\n");

        // Debug context information
        if let Ok(debug_context) = self.debug_context.lock() {
            report.push_str(&debug_context.generate_debug_summary());
        }

        report.push_str("===== END REPORT =====\n");
        report
    }

    /// Export debug information for external tools
    pub fn export_for_external_debugger(&self, format: ExportFormat) -> Result<Vec<u8>, String> {
        match format {
            ExportFormat::DWARF => {
                if let Ok(debug_context) = self.debug_context.lock() {
                    Ok(debug_context.export_gdb_debug_info())
                } else {
                    Err("Failed to lock debug context".to_string())
                }
            }
            ExportFormat::LLDB => {
                if let Ok(debug_context) = self.debug_context.lock() {
                    Ok(debug_context.export_lldb_debug_info())
                } else {
                    Err("Failed to lock debug context".to_string())
                }
            }
            ExportFormat::JSON => {
                let report = self.generate_debug_report();
                Ok(report.into_bytes())
            }
            ExportFormat::XML => {
                let xml_data = self.generate_xml_debug_info()?;
                Ok(xml_data.into_bytes())
            }
        }
    }

    // Backend-specific implementations
    fn set_gdb_breakpoint(&self, breakpoint: &DebugBreakpoint) -> Result<(), String> {
        // Implementation for GDB breakpoint setting
        let gdb_command = format!(
            "break {}:{}",
            breakpoint.location.file, breakpoint.location.line
        );
        self.execute_gdb_command(&gdb_command).map(|_| ())
    }

    fn set_lldb_breakpoint(&self, breakpoint: &DebugBreakpoint) -> Result<(), String> {
        // Implementation for LLDB breakpoint setting
        let lldb_command = format!(
            "breakpoint set --file {} --line {}",
            breakpoint.location.file, breakpoint.location.line
        );
        self.execute_lldb_command(&lldb_command).map(|_| ())
    }

    fn set_mlir_breakpoint(&self, breakpoint: &DebugBreakpoint) -> Result<(), String> {
        // Implementation for MLIR debugger breakpoint setting
        let mlir_command = format!(
            "mlir-debug break {}:{}:{}",
            breakpoint.location.file, breakpoint.location.line, breakpoint.location.column
        );
        self.execute_mlir_command(&mlir_command).map(|_| ())
    }

    fn set_custom_breakpoint(&self, breakpoint: &DebugBreakpoint) -> Result<(), String> {
        // Implementation for custom debugger breakpoint setting
        if let DebugBackend::Custom { command, args, .. } = &self.backend {
            let full_command = format!(
                "{} {} break {}:{}",
                command,
                args.join(" "),
                breakpoint.location.file,
                breakpoint.location.line
            );
            self.execute_custom_command(&full_command).map(|_| ())
        } else {
            Err("Not a custom backend".to_string())
        }
    }

    fn step_gdb(&self) -> Result<(), String> {
        self.execute_gdb_command("step").map(|_| ())
    }

    fn step_lldb(&self) -> Result<(), String> {
        self.execute_lldb_command("thread step-inst").map(|_| ())
    }

    fn step_mlir_debugger(&self) -> Result<(), String> {
        self.execute_mlir_command("step").map(|_| ())
    }

    fn step_custom(&self) -> Result<(), String> {
        if let DebugBackend::Custom { command, args, .. } = &self.backend {
            let full_command = format!("{} {} step", command, args.join(" "));
            self.execute_custom_command(&full_command).map(|_| ())
        } else {
            Err("Not a custom backend".to_string())
        }
    }

    fn get_gdb_variable_value(
        &self,
        var_name: &str,
        _symbol: &VariableSymbol,
    ) -> Result<VariableValue, String> {
        // Execute GDB print command and parse result
        let output = self.execute_gdb_command(&format!("print {}", var_name))?;
        self.parse_gdb_variable_output(&output)
    }

    fn get_lldb_variable_value(
        &self,
        var_name: &str,
        _symbol: &VariableSymbol,
    ) -> Result<VariableValue, String> {
        // Execute LLDB frame variable command and parse result
        let output = self.execute_lldb_command(&format!("frame variable {}", var_name))?;
        self.parse_lldb_variable_output(&output)
    }

    fn get_mlir_variable_value(
        &self,
        var_name: &str,
        symbol: &VariableSymbol,
    ) -> Result<VariableValue, String> {
        // For MLIR debugger, we can access the debug context directly
        if let Ok(debug_context) = self.debug_context.lock() {
            // Simulate variable value retrieval
            Ok(VariableValue {
                type_name: symbol.type_name.clone(),
                value: format!("mlir_value_{}", var_name),
                is_optimized_out: false,
                children: None,
            })
        } else {
            Err("Failed to lock debug context".to_string())
        }
    }

    fn get_custom_variable_value(
        &self,
        var_name: &str,
        symbol: &VariableSymbol,
    ) -> Result<VariableValue, String> {
        // Implementation for custom debugger variable retrieval
        Ok(VariableValue {
            type_name: symbol.type_name.clone(),
            value: format!("custom_value_{}", var_name),
            is_optimized_out: false,
            children: None,
        })
    }

    // Utility methods
    fn execute_gdb_command(&self, command: &str) -> Result<String, String> {
        // Execute GDB command and return output
        let output = Command::new("gdb")
            .args(&["--batch", "--ex", command])
            .output()
            .map_err(|e| format!("Failed to execute GDB command: {}", e))?;

        String::from_utf8(output.stdout).map_err(|e| format!("Failed to parse GDB output: {}", e))
    }

    fn execute_lldb_command(&self, command: &str) -> Result<String, String> {
        // Execute LLDB command and return output
        let output = Command::new("lldb")
            .args(&["--batch", "--one-line", command])
            .output()
            .map_err(|e| format!("Failed to execute LLDB command: {}", e))?;

        String::from_utf8(output.stdout).map_err(|e| format!("Failed to parse LLDB output: {}", e))
    }

    fn execute_mlir_command(&self, command: &str) -> Result<String, String> {
        // Execute MLIR debugger command
        if let DebugBackend::MlirDebugger {
            mlir_opt_path,
            debug_options,
        } = &self.backend
        {
            let mut cmd = Command::new(mlir_opt_path);
            cmd.args(debug_options);
            cmd.args(&["--debug-command", command]);

            let output = cmd
                .output()
                .map_err(|e| format!("Failed to execute MLIR command: {}", e))?;

            String::from_utf8(output.stdout)
                .map_err(|e| format!("Failed to parse MLIR output: {}", e))
        } else {
            Err("Not an MLIR debugger backend".to_string())
        }
    }

    fn execute_custom_command(&self, command: &str) -> Result<String, String> {
        // Execute custom debugger command
        let output = Command::new("sh")
            .args(&["-c", command])
            .output()
            .map_err(|e| format!("Failed to execute custom command: {}", e))?;

        String::from_utf8(output.stdout)
            .map_err(|e| format!("Failed to parse custom output: {}", e))
    }

    fn parse_gdb_variable_output(&self, output: &str) -> Result<VariableValue, String> {
        // Parse GDB variable output format
        // Example: "$1 = 42"
        if let Some(equals_pos) = output.find('=') {
            let value_part = output[equals_pos + 1..].trim();
            Ok(VariableValue {
                type_name: "unknown".to_string(),
                value: value_part.to_string(),
                is_optimized_out: value_part.contains("optimized out"),
                children: None,
            })
        } else {
            Err("Invalid GDB output format".to_string())
        }
    }

    fn parse_lldb_variable_output(&self, output: &str) -> Result<VariableValue, String> {
        // Parse LLDB variable output format
        // Example: "(int) var_name = 42"
        let parts: Vec<&str> = output.split('=').collect();
        if parts.len() >= 2 {
            let value = parts[1].trim();
            let type_info = parts[0].trim();

            // Extract type from parentheses
            let type_name = if let Some(start) = type_info.find('(') {
                if let Some(end) = type_info.find(')') {
                    type_info[start + 1..end].to_string()
                } else {
                    "unknown".to_string()
                }
            } else {
                "unknown".to_string()
            };

            Ok(VariableValue {
                type_name,
                value: value.to_string(),
                is_optimized_out: value.contains("optimized out"),
                children: None,
            })
        } else {
            Err("Invalid LLDB output format".to_string())
        }
    }

    fn build_symbol_table(&mut self, _executable_path: &str) -> Result<(), String> {
        // Build symbol table from debug information
        // This would typically parse DWARF or other debug formats

        // For now, populate with basic symbols from debug context
        if let Ok(debug_context) = self.debug_context.lock() {
            for (var_id, var_info) in &debug_context.variables {
                let symbol = VariableSymbol {
                    name: var_info.name.clone(),
                    type_name: var_info.type_info.mlir_type.clone(),
                    location: var_info.location.clone(),
                    scope: var_info.scope.clone(),
                    storage_location: StorageLocation::Register(format!("reg_{}", var_id.0)),
                };

                self.symbol_table
                    .variables
                    .insert(var_info.name.clone(), symbol);
            }
        }

        Ok(())
    }

    fn update_execution_state(&mut self) -> Result<(), String> {
        // Update execution state based on debugger backend
        // This would typically query the debugger for current state

        // For now, simulate stepping through locations
        if let Ok(debug_context) = self.debug_context.lock() {
            if !debug_context.debug_locations.is_empty() {
                let location = debug_context.debug_locations[0].clone();
                self.execution_state = ExecutionState::Paused {
                    location,
                    reason: PauseReason::StepComplete,
                };
            }
        }

        Ok(())
    }

    fn evaluate_watch_expressions(&mut self) -> Result<(), String> {
        // Evaluate all active watch expressions
        for (watch_id, watch) in &mut self.watch_expressions {
            if watch.enabled {
                // Simulate expression evaluation
                let new_value = format!("eval_result_{}", watch_id);

                if watch.last_value.as_ref() != Some(&new_value) {
                    watch.change_count += 1;
                    watch.last_value = Some(new_value);
                }
            }
        }

        Ok(())
    }

    fn generate_xml_debug_info(&self) -> Result<String, String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<mlir_debug_info>\n");

        xml.push_str("  <breakpoints>\n");
        for (id, bp) in &self.active_breakpoints {
            xml.push_str(&format!(
                "    <breakpoint id=\"{}\" file=\"{}\" line=\"{}\" column=\"{}\" enabled=\"{}\" hits=\"{}\"/>\n",
                id, bp.location.file, bp.location.line, bp.location.column, bp.enabled, bp.hit_count
            ));
        }
        xml.push_str("  </breakpoints>\n");

        xml.push_str("  <watch_expressions>\n");
        for (id, watch) in &self.watch_expressions {
            xml.push_str(&format!(
                "    <watch id=\"{}\" expression=\"{}\" enabled=\"{}\" changes=\"{}\"/>\n",
                id, watch.expression, watch.enabled, watch.change_count
            ));
        }
        xml.push_str("  </watch_expressions>\n");

        xml.push_str("</mlir_debug_info>\n");
        Ok(xml)
    }

    fn log_debug_event(
        &mut self,
        level: LogLevel,
        message: &str,
        location: Option<UniversalDebugLocation>,
    ) {
        if let Some(ref mut session) = self.debug_session {
            session.debug_log.push(DebugLogEntry {
                timestamp: std::time::SystemTime::now(),
                level,
                message: message.to_string(),
                location,
            });
        }
    }
}

pub enum ExportFormat {
    DWARF,
    LLDB,
    JSON,
    XML,
}

impl SymbolTable {
    fn new() -> Self {
        Self {
            functions: HashMap::new(),
            variables: HashMap::new(),
            types: HashMap::new(),
            modules: HashMap::new(),
        }
    }
}

/// Factory function to create debugger with appropriate backend
pub fn create_mlir_debugger(
    debug_context: UniversalMlirDebugContext,
    backend_type: &str,
) -> Result<MlirDebugger, String> {
    let backend = match backend_type {
        "gdb" => DebugBackend::GDB {
            executable_path: "gdb".to_string(),
            args: vec!["--quiet".to_string(), "--batch".to_string()],
        },
        "lldb" => DebugBackend::LLDB {
            executable_path: "lldb".to_string(),
            args: vec!["--batch".to_string()],
        },
        "mlir" => DebugBackend::MlirDebugger {
            mlir_opt_path: "mlir-opt".to_string(),
            debug_options: vec!["--debug".to_string(), "--debug-only=mlir".to_string()],
        },
        _ => return Err(format!("Unknown backend type: {}", backend_type)),
    };

    Ok(MlirDebugger::new(debug_context, backend))
}
