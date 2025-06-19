use ptx::checkpoint_integration::{compile_ptx_with_auto_checkpoint, CheckpointedCompiler};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 解析命令行参数
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    let command = &args[1];

    match command.as_str() {
        "compile" => {
            if args.len() < 3 {
                eprintln!("错误: 需要提供PTX文件路径");
                print_usage();
                return Ok(());
            }

            let ptx_file = &args[2];
            let checkpoint_dir = args
                .get(3)
                .map(|s| PathBuf::from(s))
                .unwrap_or_else(|| PathBuf::from("./checkpoints"));

            compile_command(ptx_file, &checkpoint_dir)?;
        }
        "resume" => {
            if args.len() < 3 {
                eprintln!("错误: 需要提供检查点ID");
                print_usage();
                return Ok(());
            }

            let checkpoint_id = &args[2];
            let checkpoint_dir = args
                .get(3)
                .map(|s| PathBuf::from(s))
                .unwrap_or_else(|| PathBuf::from("./checkpoints"));

            resume_command(checkpoint_id, &checkpoint_dir)?;
        }
        "list" => {
            let checkpoint_dir = args
                .get(2)
                .map(|s| PathBuf::from(s))
                .unwrap_or_else(|| PathBuf::from("./checkpoints"));

            list_command(&checkpoint_dir)?;
        }
        "report" => {
            let checkpoint_dir = args
                .get(2)
                .map(|s| PathBuf::from(s))
                .unwrap_or_else(|| PathBuf::from("./checkpoints"));

            report_command(&checkpoint_dir)?;
        }
        "cleanup" => {
            let checkpoint_dir = args
                .get(2)
                .map(|s| PathBuf::from(s))
                .unwrap_or_else(|| PathBuf::from("./checkpoints"));

            cleanup_command(&checkpoint_dir)?;
        }
        "demo" => {
            demo_command()?;
        }
        _ => {
            eprintln!("未知命令: {}", command);
            print_usage();
        }
    }

    Ok(())
}

fn print_usage() {
    println!("ZLUDA PTX 检查点系统演示");
    println!();
    println!("用法:");
    println!("  cargo run --example checkpoint_demo compile <ptx_file> [checkpoint_dir]");
    println!("  cargo run --example checkpoint_demo resume <checkpoint_id> [checkpoint_dir]");
    println!("  cargo run --example checkpoint_demo list [checkpoint_dir]");
    println!("  cargo run --example checkpoint_demo report [checkpoint_dir]");
    println!("  cargo run --example checkpoint_demo cleanup [checkpoint_dir]");
    println!("  cargo run --example checkpoint_demo demo");
    println!();
    println!("命令:");
    println!("  compile    编译PTX文件并创建检查点");
    println!("  resume     从检查点恢复编译");
    println!("  list       列出所有检查点");
    println!("  report     生成编译报告");
    println!("  cleanup    清理旧检查点");
    println!("  demo       运行内置演示");
}

fn compile_command(
    ptx_file: &str,
    checkpoint_dir: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 开始编译PTX文件: {}", ptx_file);
    println!("📁 检查点目录: {:?}", checkpoint_dir);

    // 读取PTX文件
    let ptx_content = fs::read_to_string(ptx_file)
        .map_err(|e| format!("无法读取PTX文件 '{}': {}", ptx_file, e))?;

    println!("📄 PTX文件大小: {} 字节", ptx_content.len());

    // 使用检查点编译器
    let mut compiler = CheckpointedCompiler::new(checkpoint_dir, true)?;

    let description = format!("编译文件: {}", ptx_file);
    match compiler.compile_ptx_with_checkpoints(&ptx_content, Some(description)) {
        Ok(result) => {
            println!("✅ 编译成功!");
            println!("🆔 检查点ID: {}", result.checkpoint_id);
            println!("⏱️  编译时间: {}ms", result.compilation_time_ms);

            if let Some(spirv) = result.spirv_binary {
                println!("📦 SPIR-V大小: {} 字节", spirv.len());
            }

            if !result.errors.is_empty() {
                println!("⚠️  警告信息:");
                for error in result.errors {
                    println!("   - {}", error);
                }
            }
        }
        Err(e) => {
            eprintln!("❌ 编译失败: {}", e);
        }
    }

    Ok(())
}

fn resume_command(
    checkpoint_id: &str,
    checkpoint_dir: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 从检查点恢复编译: {}", checkpoint_id);

    let mut compiler = CheckpointedCompiler::new(checkpoint_dir, true)?;

    match compiler.resume_from_checkpoint(checkpoint_id) {
        Ok(result) => {
            println!("✅ 恢复成功!");
            println!("⏱️  编译时间: {}ms", result.compilation_time_ms);
        }
        Err(e) => {
            eprintln!("❌ 恢复失败: {}", e);
        }
    }

    Ok(())
}

fn list_command(checkpoint_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 检查点列表 (目录: {:?})", checkpoint_dir);

    let compiler = CheckpointedCompiler::new(checkpoint_dir, false)?;
    let checkpoints = compiler.list_checkpoints();

    if checkpoints.is_empty() {
        println!("  (无检查点)");
        return Ok(());
    }

    println!();
    for checkpoint in checkpoints {
        let timestamp =
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(checkpoint.timestamp);
        let datetime = humantime::format_rfc3339(timestamp);

        println!("🔹 ID: {}", checkpoint.id);
        println!("   描述: {}", checkpoint.description);
        println!("   时间: {}", datetime);
        println!("   位置: {}", checkpoint.created_at);
        println!();
    }

    Ok(())
}

fn report_command(checkpoint_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 生成编译报告 (目录: {:?})", checkpoint_dir);

    let compiler = CheckpointedCompiler::new(checkpoint_dir, false)?;
    let report = compiler.generate_compilation_report();

    println!();
    println!("{}", report);

    Ok(())
}

fn cleanup_command(checkpoint_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧹 清理旧检查点 (目录: {:?})", checkpoint_dir);

    let mut compiler = CheckpointedCompiler::new(checkpoint_dir, false)?;

    match compiler.cleanup_checkpoints() {
        Ok(count) => {
            println!("✅ 清理完成，删除了 {} 个检查点", count);
        }
        Err(e) => {
            eprintln!("❌ 清理失败: {}", e);
        }
    }

    Ok(())
}

fn demo_command() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 ZLUDA PTX 检查点系统演示");
    println!();

    // 创建临时目录
    let temp_dir = tempfile::TempDir::new()?;
    let checkpoint_dir = temp_dir.path();

    println!("📁 使用临时检查点目录: {:?}", checkpoint_dir);

    // 演示PTX代码
    let demo_ptx_sources = vec![
        (
            "简单内核",
            r#"
.version 7.0
.target sm_50
.address_size 64

.entry simple_kernel() {
    ret;
}
"#,
        ),
        (
            "向量加法",
            r#"
.version 7.0
.target sm_50
.address_size 64

.entry vector_add(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n
) {
    .reg .u32 %tid;
    .reg .u64 %a_addr, %b_addr, %c_addr;
    .reg .f32 %a_val, %b_val, %c_val;
    
    mov.u32 %tid, %ctaid.x;
    
    mul.wide.u32 %a_addr, %tid, 4;
    ld.param.u64 %a_addr, [a_ptr];
    add.u64 %a_addr, %a_addr, %a_addr;
    
    ld.global.f32 %a_val, [%a_addr];
    
    ret;
}
"#,
        ),
        (
            "错误示例", // 这个会失败
            r#"
.version 7.0
.target sm_50
.address_size 64

.entry bad_kernel() {
    // 故意的语法错误
    invalid_instruction;
}
"#,
        ),
    ];

    let mut compiler = CheckpointedCompiler::new(checkpoint_dir, true)?;

    // 编译所有演示代码
    for (name, ptx_source) in demo_ptx_sources {
        println!("\n🔨 编译演示: {}", name);
        println!("{'=':<50}");

        match compiler.compile_ptx_with_checkpoints(ptx_source, Some(name.to_string())) {
            Ok(result) => {
                println!("✅ 编译成功 - 检查点: {}", result.checkpoint_id);
            }
            Err(e) => {
                println!("❌ 编译失败: {}", e);
            }
        }
    }

    // 显示检查点列表
    println!("\n📋 创建的检查点:");
    println!("{'=':<50}");
    for checkpoint in compiler.list_checkpoints() {
        println!("🔹 {}: {}", checkpoint.id, checkpoint.description);
    }

    // 生成报告
    println!("\n📊 编译报告:");
    println!("{'=':<50}");
    println!("{}", compiler.generate_compilation_report());

    // 演示检查点恢复
    if let Some(checkpoint) = compiler.list_checkpoints().first() {
        println!("\n🔄 演示检查点恢复:");
        println!("{'=':<50}");

        match compiler.resume_from_checkpoint(&checkpoint.id) {
            Ok(_) => {
                println!("✅ 检查点恢复成功");
            }
            Err(e) => {
                println!("❌ 检查点恢复失败: {}", e);
            }
        }
    }

    println!("\n🎉 演示完成!");

    Ok(())
}

// 添加humantime依赖的简单实现
mod humantime {
    use std::time::SystemTime;

    pub fn format_rfc3339(time: SystemTime) -> String {
        let duration = time.duration_since(std::time::UNIX_EPOCH).unwrap();
        let secs = duration.as_secs();

        // 简单的时间格式化
        let days = secs / 86400;
        let hours = (secs % 86400) / 3600;
        let minutes = (secs % 3600) / 60;
        let seconds = secs % 60;

        if days > 0 {
            format!("{}天{}小时前", days, hours)
        } else if hours > 0 {
            format!("{}小时{}分钟前", hours, minutes)
        } else if minutes > 0 {
            format!("{}分钟前", minutes)
        } else {
            format!("{}秒前", seconds)
        }
    }
}
