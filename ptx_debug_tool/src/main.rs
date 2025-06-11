// PTX Debug Tool - State Recovery CLI
// This tool provides command-line interface for PTX debugging and state recovery

use clap::{Args, Parser, Subcommand};
use ptx::debug::*;
use ptx::state_recovery::*;
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ptx-debug")]
#[command(about = "PTX Debug Tool for state recovery and debugging")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile PTX with debug information
    Compile(CompileArgs),
    /// Analyze debug mappings
    Analyze(AnalyzeArgs),
    /// Interactive debugging session
    Debug(DebugArgs),
    /// Recover state from crash dump
    Recover(RecoverArgs),
    /// Export debug information
    Export(ExportArgs),
}

#[derive(Args)]
struct CompileArgs {
    /// Input PTX file
    #[arg(short, long)]
    input: PathBuf,

    /// Output debug mapping file
    #[arg(short, long)]
    output: PathBuf,

    /// Target architecture (amd_gcn, intel_spirv, sass)
    #[arg(short, long, default_value = "amd_gcn")]
    target: String,

    /// Enable verbose debug output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Args)]
struct AnalyzeArgs {
    /// Debug mapping file to analyze
    #[arg(short, long)]
    mapping_file: PathBuf,

    /// Show detailed instruction mappings
    #[arg(short, long)]
    detailed: bool,

    /// Filter by PTX line number
    #[arg(short, long)]
    line: Option<u32>,
}

#[derive(Args)]
struct DebugArgs {
    /// Debug mapping file
    #[arg(short, long)]
    mapping_file: PathBuf,

    /// Initial breakpoint location (file:line:column)
    #[arg(short, long)]
    breakpoint: Option<String>,

    /// GDB-compatible mode
    #[arg(short, long)]
    gdb_mode: bool,
}

#[derive(Args)]
struct RecoverArgs {
    /// Debug mapping file
    #[arg(short, long)]
    mapping_file: PathBuf,

    /// Target architecture crash address
    #[arg(short, long)]
    address: String,

    /// Target architecture (amd_gcn, intel_spirv, sass)
    #[arg(short, long, default_value = "amd_gcn")]
    target: String,

    /// Memory dump file
    #[arg(long)]
    memory_dump: Option<PathBuf>,
}

#[derive(Args)]
struct ExportArgs {
    /// Debug mapping file
    #[arg(short, long)]
    mapping_file: PathBuf,

    /// Export format (json, gdb, vscode)
    #[arg(short, long, default_value = "json")]
    format: String,

    /// Output file
    #[arg(short, long)]
    output: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile(args) => compile_command(args).await,
        Commands::Analyze(args) => analyze_command(args).await,
        Commands::Debug(args) => debug_command(args).await,
        Commands::Recover(args) => recover_command(args).await,
        Commands::Export(args) => export_command(args).await,
    }
}

async fn compile_command(args: CompileArgs) -> anyhow::Result<()> {
    println!("Compiling PTX with debug information...");
    println!("Input: {}", args.input.display());
    println!("Output: {}", args.output.display());
    println!("Target: {}", args.target);

    // Read PTX source
    let ptx_content = fs::read_to_string(&args.input)?;

    // Compile PTX to LLVM IR with debug information
    let result = ptx::ptx_to_llvm_with_debug(&ptx_content);

    match result {
        Ok((llvm_module, debug_mappings)) => {
            // Convert LLVM module to string
            let llvm_ir = llvm_module
                .print_to_string()
                .map_err(|e| anyhow::anyhow!("Failed to convert LLVM module to string: {}", e))?;

            // Write LLVM IR to output file
            fs::write(&args.output, llvm_ir)?;
            println!(
                "LLVM IR with debug info saved to: {}",
                args.output.display()
            );

            if args.verbose {
                println!("Debug mappings: {} entries", debug_mappings.len());
                for (i, mapping) in debug_mappings.iter().enumerate().take(5) {
                    println!(
                        "  [{}] {}:{}:{} -> {} instructions",
                        i,
                        mapping.ptx_location.file,
                        mapping.ptx_location.line,
                        mapping.ptx_location.column,
                        mapping.target_instructions.len()
                    );
                }
                if debug_mappings.len() > 5 {
                    println!("  ... and {} more entries", debug_mappings.len() - 5);
                }
            }
        }
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            return Err(anyhow::anyhow!("PTX compilation failed: {}", e));
        }
    }

    Ok(())
}

async fn analyze_command(args: AnalyzeArgs) -> anyhow::Result<()> {
    println!("Analyzing debug mappings...");

    let content = fs::read_to_string(&args.mapping_file)?;
    let mappings: Vec<DwarfMappingEntry> = serde_json::from_str(&content)?;

    println!("Total mappings: {}", mappings.len());

    for (i, mapping) in mappings.iter().enumerate() {
        if let Some(line_filter) = args.line {
            if mapping.ptx_location.line != line_filter {
                continue;
            }
        }

        println!("\nMapping {}:", i + 1);
        println!(
            "  PTX Location: {}:{}:{}",
            mapping.ptx_location.file, mapping.ptx_location.line, mapping.ptx_location.column
        );

        println!(
            "  Target Instructions: {}",
            mapping.target_instructions.len()
        );

        if args.detailed {
            for (j, inst) in mapping.target_instructions.iter().enumerate() {
                match inst {
                    TargetInstruction::AmdGcn {
                        instruction,
                        address,
                        ..
                    } => {
                        println!("    [{}] AMD GCN: {} @ 0x{:x}", j, instruction, address);
                    }
                    TargetInstruction::IntelSpirv {
                        instruction,
                        opcode,
                        ..
                    } => {
                        println!("    [{}] SPIRV: {} (opcode: {})", j, instruction, opcode);
                    }
                    TargetInstruction::Sass {
                        instruction,
                        address,
                        predicate,
                    } => {
                        let pred_str = predicate
                            .as_ref()
                            .map(|p| format!(" [{}]", p))
                            .unwrap_or_default();
                        println!(
                            "    [{}] SASS: {} @ 0x{:x}{}",
                            j, instruction, address, pred_str
                        );
                    }
                }
            }
        }

        if !mapping.variable_mappings.is_empty() {
            println!("  Variables: {}", mapping.variable_mappings.len());
            if args.detailed {
                for (name, location) in &mapping.variable_mappings {
                    println!("    {} -> {:?}", name, location);
                }
            }
        }
    }

    Ok(())
}

async fn debug_command(args: DebugArgs) -> anyhow::Result<()> {
    println!("Starting interactive debugging session...");

    let mut manager = PtxStateRecoveryManager::new();
    manager.load_debug_mappings(&args.mapping_file).unwrap();

    // Set initial breakpoint if provided
    if let Some(bp_location) = args.breakpoint {
        let parts: Vec<&str> = bp_location.split(':').collect();
        if parts.len() >= 2 {
            let file = parts[0].to_string();
            let line = parts[1].parse::<u32>().unwrap_or(1);
            let column = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

            let location = PtxSourceLocation {
                file,
                line,
                column,
                instruction_offset: 0,
            };

            let bp_id = manager.add_breakpoint(location, None);
            println!(
                "Breakpoint {} set at {}:{}:{}",
                bp_id, parts[0], line, column
            );
        }
    }

    // Interactive debugging loop
    loop {
        print!("(ptx-debug) ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input {
            "quit" | "q" | "exit" => {
                println!("Goodbye!");
                break;
            }
            "help" | "h" => {
                print_debug_help();
            }
            "info" | "i" => {
                println!("{}", manager.generate_state_dump());
            }
            "backtrace" | "bt" => {
                println!("Call stack:");
                for (i, frame) in manager.get_call_stack().iter().enumerate() {
                    println!(
                        "  #{}: {} at {}:{}:{}",
                        i,
                        frame.function_name,
                        frame.location.file,
                        frame.location.line,
                        frame.location.column
                    );
                }
            }
            cmd if cmd.starts_with("break ") || cmd.starts_with("b ") => {
                let location_str = cmd.split_whitespace().nth(1).unwrap_or("");
                let parts: Vec<&str> = location_str.split(':').collect();
                if parts.len() >= 2 {
                    let file = parts[0].to_string();
                    let line = parts[1].parse::<u32>().unwrap_or(1);
                    let column = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

                    let location = PtxSourceLocation {
                        file: file.clone(),
                        line,
                        column,
                        instruction_offset: 0,
                    };

                    let bp_id = manager.add_breakpoint(location, None);
                    println!("Breakpoint {} set at {}:{}:{}", bp_id, file, line, column);
                } else {
                    println!("Usage: break <file>:<line>[:<column>]");
                }
            }
            cmd if cmd.starts_with("delete ") || cmd.starts_with("d ") => {
                if let Some(id_str) = cmd.split_whitespace().nth(1) {
                    if let Ok(id) = id_str.parse::<u32>() {
                        if manager.remove_breakpoint(id) {
                            println!("Breakpoint {} deleted", id);
                        } else {
                            println!("No breakpoint with id {}", id);
                        }
                    } else {
                        println!("Invalid breakpoint id: {}", id_str);
                    }
                } else {
                    println!("Usage: delete <breakpoint_id>");
                }
            }
            cmd if cmd.starts_with("print ") || cmd.starts_with("p ") => {
                if let Some(var_name) = cmd.split_whitespace().nth(1) {
                    if let Some(value) = manager.get_variable_value(var_name) {
                        println!("{} = {:?}", var_name, value);
                    } else {
                        println!("Variable '{}' not found", var_name);
                    }
                } else {
                    println!("Usage: print <variable_name>");
                }
            }
            cmd if cmd.starts_with("recover ") => {
                if let Some(addr_str) = cmd.split_whitespace().nth(1) {
                    if let Ok(address) = u64::from_str_radix(addr_str.trim_start_matches("0x"), 16)
                    {
                        if let Some(location) =
                            manager.find_ptx_location_from_target("amd_gcn", address)
                        {
                            println!(
                                "Address 0x{:x} maps to PTX location: {}:{}:{}",
                                address, location.file, location.line, location.column
                            );
                        } else {
                            println!("No PTX location found for address 0x{:x}", address);
                        }
                    } else {
                        println!("Invalid address format: {}", addr_str);
                    }
                } else {
                    println!("Usage: recover <hex_address>");
                }
            }
            "" => {
                // Empty line, continue
            }
            _ => {
                println!(
                    "Unknown command: {}. Type 'help' for available commands.",
                    input
                );
            }
        }
    }

    Ok(())
}

async fn recover_command(args: RecoverArgs) -> anyhow::Result<()> {
    println!("Recovering PTX state from crash...");

    let mut manager = PtxStateRecoveryManager::new();
    manager.load_debug_mappings(&args.mapping_file).unwrap();

    // Parse address
    let address = if args.address.starts_with("0x") {
        u64::from_str_radix(&args.address[2..], 16)?
    } else {
        args.address.parse::<u64>()?
    };

    // Find PTX location
    if let Some(location) = manager.find_ptx_location_from_target(&args.target, address) {
        println!("Crash occurred at PTX location:");
        println!("  File: {}", location.file);
        println!("  Line: {}", location.line);
        println!("  Column: {}", location.column);
        println!("  Instruction Offset: {}", location.instruction_offset);

        // Load memory dump if provided
        if let Some(memory_dump_path) = args.memory_dump {
            let memory_data = fs::read(&memory_dump_path)?;
            manager.take_memory_snapshot("crash_dump".to_string(), address, memory_data);
        }

        // Generate recovery report
        println!("\n{}", manager.generate_state_dump());
    } else {
        println!(
            "No PTX location found for address 0x{:x} in {} architecture",
            address, args.target
        );
    }

    Ok(())
}

async fn export_command(args: ExportArgs) -> anyhow::Result<()> {
    println!("Exporting debug information...");

    let manager = PtxStateRecoveryManager::new();

    match args.format.as_str() {
        "json" => {
            let content = fs::read_to_string(&args.mapping_file)?;
            fs::write(&args.output, content)?;
            println!("JSON export saved to: {}", args.output.display());
        }
        "gdb" => {
            let gdb_info = manager.export_gdb_compatible_info();
            fs::write(&args.output, gdb_info)?;
            println!("GDB-compatible export saved to: {}", args.output.display());
        }
        "vscode" => {
            // Create VS Code debug configuration
            let vscode_config = serde_json::json!({
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": "PTX Debug",
                        "type": "gdb",
                        "request": "attach",
                        "program": "${workspaceFolder}/target/debug/main",
                        "MIMode": "gdb",
                        "setupCommands": [
                            {
                                "description": "Load PTX debug symbols",
                                "text": "source ptx_debug.gdb",
                                "ignoreFailures": false
                            }
                        ]
                    }
                ]
            });
            fs::write(&args.output, serde_json::to_string_pretty(&vscode_config)?)?;
            println!(
                "VS Code debug configuration saved to: {}",
                args.output.display()
            );
        }
        _ => {
            println!("Unknown export format: {}", args.format);
            println!("Supported formats: json, gdb, vscode");
        }
    }

    Ok(())
}

fn print_debug_help() {
    println!("Available commands:");
    println!("  help, h                 - Show this help");
    println!("  info, i                 - Show current state information");
    println!("  backtrace, bt           - Show call stack");
    println!("  break <loc>, b <loc>    - Set breakpoint at location (file:line:column)");
    println!("  delete <id>, d <id>     - Delete breakpoint by id");
    println!("  print <var>, p <var>    - Print variable value");
    println!("  recover <addr>          - Recover PTX location from target address");
    println!("  quit, q, exit           - Exit debugger");
    println!();
    println!("Examples:");
    println!("  break kernel.ptx:42:10  - Set breakpoint at line 42, column 10");
    println!("  print tid               - Print value of variable 'tid'");
    println!("  recover 0x1000          - Find PTX location for address 0x1000");
}
