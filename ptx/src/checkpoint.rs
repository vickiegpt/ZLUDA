use crate::debug::{DwarfMappingEntry, PtxSourceLocation};
use crate::TranslateError;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use rand::random;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// 检查点数据结构，包含编译过程中的所有重要状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationCheckpoint {
    /// 检查点元数据
    pub metadata: CheckpointMetadata,
    /// PTX源代码
    pub ptx_source: String,
    /// LLVM IR代码
    pub llvm_ir: Option<String>,
    /// SPIR-V二进制数据（base64编码）
    pub spirv_binary: Option<String>,
    /// 编译阶段信息
    pub compilation_stage: CompilationStage,
    /// 错误信息（如果有的话）
    pub error_info: Option<ErrorInfo>,
    /// 调试映射信息
    pub debug_mappings: Vec<DwarfMappingEntry>,
    /// PTX源码位置映射
    pub source_mappings: HashMap<u64, PtxSourceLocation>,
    /// 编译选项和参数
    pub compile_options: CompileOptions,
    /// 性能统计信息
    pub performance_stats: PerformanceStats,
}

/// 检查点元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// 检查点ID
    pub id: String,
    /// 创建时间戳
    pub timestamp: u64,
    /// ZLUDA版本
    pub zluda_version: String,
    /// 创建位置（文件名:行号）
    pub created_at: String,
    /// 检查点描述
    pub description: String,
    /// 关联的源文件路径
    pub source_file_path: Option<PathBuf>,
}

/// 编译阶段枚举
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CompilationStage {
    /// PTX解析阶段
    PtxParsing,
    /// LLVM IR生成阶段
    LlvmGeneration,
    /// LLVM优化阶段
    LlvmOptimization,
    /// SPIR-V转换阶段
    SpirvConversion,
    /// 调试信息生成阶段
    DebugGeneration,
    /// 编译完成
    Completed,
    /// 编译失败
    Failed,
}

/// 错误信息结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// 错误类型
    pub error_type: String,
    /// 错误消息
    pub message: String,
    /// 错误发生的源码位置
    pub source_location: Option<PtxSourceLocation>,
    /// 调用栈信息
    pub call_stack: Vec<String>,
    /// 相关的LLVM IR片段
    pub llvm_context: Option<String>,
}

/// 编译选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileOptions {
    /// 优化级别
    pub optimization_level: u32,
    /// 是否启用调试信息
    pub debug_enabled: bool,
    /// 目标架构
    pub target_arch: String,
    /// 地址空间映射策略
    pub address_space_strategy: String,
    /// 自定义编译标志
    pub custom_flags: HashMap<String, String>,
}

/// 性能统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// PTX解析时间（毫秒）
    pub ptx_parse_time_ms: u64,
    /// LLVM生成时间（毫秒）
    pub llvm_gen_time_ms: u64,
    /// SPIR-V转换时间（毫秒）
    pub spirv_conv_time_ms: u64,
    /// 总编译时间（毫秒）
    pub total_time_ms: u64,
    /// 内存使用峰值（字节）
    pub peak_memory_bytes: u64,
    /// LLVM IR大小（字节）
    pub llvm_ir_size_bytes: usize,
    /// SPIR-V二进制大小（字节）
    pub spirv_binary_size_bytes: usize,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            ptx_parse_time_ms: 0,
            llvm_gen_time_ms: 0,
            spirv_conv_time_ms: 0,
            total_time_ms: 0,
            peak_memory_bytes: 0,
            llvm_ir_size_bytes: 0,
            spirv_binary_size_bytes: 0,
        }
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            optimization_level: 2,
            debug_enabled: true,
            target_arch: "x86_64".to_string(),
            address_space_strategy: "spirv_compatible".to_string(),
            custom_flags: HashMap::new(),
        }
    }
}

/// 检查点管理器
pub struct CheckpointManager {
    /// 检查点存储目录
    checkpoint_dir: PathBuf,
    /// 当前活动的检查点
    active_checkpoints: HashMap<String, CompilationCheckpoint>,
    /// 自动保存间隔（秒）
    auto_save_interval: u64,
    /// 最大保留检查点数量
    max_checkpoints: usize,
}

impl CheckpointManager {
    /// 创建新的检查点管理器
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> Result<Self, std::io::Error> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();
        fs::create_dir_all(&checkpoint_dir)?;

        Ok(Self {
            checkpoint_dir,
            active_checkpoints: HashMap::new(),
            auto_save_interval: 300, // 5分钟
            max_checkpoints: 100,
        })
    }

    /// 创建新的检查点
    pub fn create_checkpoint(
        &mut self,
        ptx_source: String,
        stage: CompilationStage,
        description: String,
    ) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let id = format!("checkpoint_{}_{}", timestamp, random::<u16>());

        let metadata = CheckpointMetadata {
            id: id.clone(),
            timestamp,
            zluda_version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: format!("{}:{}:{}", file!(), line!(), column!()),
            description,
            source_file_path: None,
        };

        let checkpoint = CompilationCheckpoint {
            metadata,
            ptx_source,
            llvm_ir: None,
            spirv_binary: None,
            compilation_stage: stage,
            error_info: None,
            debug_mappings: Vec::new(),
            source_mappings: HashMap::new(),
            compile_options: CompileOptions::default(),
            performance_stats: PerformanceStats::default(),
        };

        self.active_checkpoints.insert(id.clone(), checkpoint);
        id
    }

    /// 更新检查点状态
    pub fn update_checkpoint(
        &mut self,
        checkpoint_id: &str,
        stage: CompilationStage,
    ) -> Result<(), CheckpointError> {
        if let Some(checkpoint) = self.active_checkpoints.get_mut(checkpoint_id) {
            checkpoint.compilation_stage = stage;
            checkpoint.metadata.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            Ok(())
        } else {
            Err(CheckpointError::CheckpointNotFound(
                checkpoint_id.to_string(),
            ))
        }
    }

    /// 添加LLVM IR到检查点
    pub fn add_llvm_ir(
        &mut self,
        checkpoint_id: &str,
        llvm_ir: String,
    ) -> Result<(), CheckpointError> {
        if let Some(checkpoint) = self.active_checkpoints.get_mut(checkpoint_id) {
            checkpoint.performance_stats.llvm_ir_size_bytes = llvm_ir.len();
            checkpoint.llvm_ir = Some(llvm_ir);
            Ok(())
        } else {
            Err(CheckpointError::CheckpointNotFound(
                checkpoint_id.to_string(),
            ))
        }
    }

    /// 添加SPIR-V二进制到检查点
    pub fn add_spirv_binary(
        &mut self,
        checkpoint_id: &str,
        spirv_binary: Vec<u8>,
    ) -> Result<(), CheckpointError> {
        if let Some(checkpoint) = self.active_checkpoints.get_mut(checkpoint_id) {
            checkpoint.performance_stats.spirv_binary_size_bytes = spirv_binary.len();
            checkpoint.spirv_binary = Some(BASE64_STANDARD.encode(&spirv_binary));
            Ok(())
        } else {
            Err(CheckpointError::CheckpointNotFound(
                checkpoint_id.to_string(),
            ))
        }
    }

    /// 添加错误信息到检查点
    pub fn add_error(
        &mut self,
        checkpoint_id: &str,
        error: &TranslateError,
        llvm_context: Option<String>,
    ) -> Result<(), CheckpointError> {
        if let Some(checkpoint) = self.active_checkpoints.get_mut(checkpoint_id) {
            let error_info = ErrorInfo {
                error_type: format!("{:?}", error),
                message: error.to_string(),
                source_location: None,  // 可以从error中提取
                call_stack: Vec::new(), // 可以添加调用栈追踪
                llvm_context,
            };
            checkpoint.error_info = Some(error_info);
            checkpoint.compilation_stage = CompilationStage::Failed;
            Ok(())
        } else {
            Err(CheckpointError::CheckpointNotFound(
                checkpoint_id.to_string(),
            ))
        }
    }

    /// 添加调试映射信息
    pub fn add_debug_mappings(
        &mut self,
        checkpoint_id: &str,
        mappings: Vec<DwarfMappingEntry>,
    ) -> Result<(), CheckpointError> {
        if let Some(checkpoint) = self.active_checkpoints.get_mut(checkpoint_id) {
            checkpoint.debug_mappings = mappings;
            Ok(())
        } else {
            Err(CheckpointError::CheckpointNotFound(
                checkpoint_id.to_string(),
            ))
        }
    }

    /// 更新性能统计
    pub fn update_performance_stats(
        &mut self,
        checkpoint_id: &str,
        stats: PerformanceStats,
    ) -> Result<(), CheckpointError> {
        if let Some(checkpoint) = self.active_checkpoints.get_mut(checkpoint_id) {
            checkpoint.performance_stats = stats;
            Ok(())
        } else {
            Err(CheckpointError::CheckpointNotFound(
                checkpoint_id.to_string(),
            ))
        }
    }

    /// 保存检查点到磁盘
    pub fn save_checkpoint(&self, checkpoint_id: &str) -> Result<PathBuf, CheckpointError> {
        if let Some(checkpoint) = self.active_checkpoints.get(checkpoint_id) {
            let filename = format!("{}.json", checkpoint_id);
            let filepath = self.checkpoint_dir.join(filename);

            let json_data = serde_json::to_string_pretty(checkpoint)
                .map_err(|e| CheckpointError::SerializationError(e.to_string()))?;

            fs::write(&filepath, json_data).map_err(|e| CheckpointError::IoError(e.to_string()))?;

            Ok(filepath)
        } else {
            Err(CheckpointError::CheckpointNotFound(
                checkpoint_id.to_string(),
            ))
        }
    }

    /// 从磁盘加载检查点
    pub fn load_checkpoint<P: AsRef<Path>>(
        &mut self,
        filepath: P,
    ) -> Result<String, CheckpointError> {
        let content = fs::read_to_string(filepath.as_ref())
            .map_err(|e| CheckpointError::IoError(e.to_string()))?;

        let checkpoint: CompilationCheckpoint = serde_json::from_str(&content)
            .map_err(|e| CheckpointError::DeserializationError(e.to_string()))?;

        let id = checkpoint.metadata.id.clone();
        self.active_checkpoints.insert(id.clone(), checkpoint);

        Ok(id)
    }

    /// 获取检查点信息
    pub fn get_checkpoint(&self, checkpoint_id: &str) -> Option<&CompilationCheckpoint> {
        self.active_checkpoints.get(checkpoint_id)
    }

    /// 列出所有活动检查点
    pub fn list_active_checkpoints(&self) -> Vec<&CheckpointMetadata> {
        self.active_checkpoints
            .values()
            .map(|cp| &cp.metadata)
            .collect()
    }

    /// 清理旧的检查点
    pub fn cleanup_old_checkpoints(&mut self) -> Result<usize, CheckpointError> {
        // 先收集需要删除的检查点ID
        let mut checkpoint_ids_with_timestamps: Vec<_> = self
            .active_checkpoints
            .values()
            .map(|cp| (cp.metadata.id.clone(), cp.metadata.timestamp))
            .collect();

        checkpoint_ids_with_timestamps.sort_by_key(|(_, timestamp)| *timestamp);

        let mut removed_count = 0;

        // 保留最新的检查点，删除超过限制的旧检查点
        if checkpoint_ids_with_timestamps.len() > self.max_checkpoints {
            let to_remove = checkpoint_ids_with_timestamps.len() - self.max_checkpoints;
            for (id, _) in checkpoint_ids_with_timestamps.iter().take(to_remove) {
                self.active_checkpoints.remove(id);

                // 同时删除磁盘文件
                let filename = format!("{}.json", id);
                let filepath = self.checkpoint_dir.join(filename);
                if filepath.exists() {
                    let _ = fs::remove_file(filepath);
                }

                removed_count += 1;
            }
        }

        Ok(removed_count)
    }

    /// 恢复编译状态
    pub fn restore_compilation_state(
        &self,
        checkpoint_id: &str,
    ) -> Result<CompilationRestoreInfo, CheckpointError> {
        if let Some(checkpoint) = self.active_checkpoints.get(checkpoint_id) {
            let spirv_binary = if let Some(ref encoded) = checkpoint.spirv_binary {
                Some(
                    BASE64_STANDARD
                        .decode(encoded)
                        .map_err(|e| CheckpointError::DeserializationError(e.to_string()))?,
                )
            } else {
                None
            };

            Ok(CompilationRestoreInfo {
                ptx_source: checkpoint.ptx_source.clone(),
                llvm_ir: checkpoint.llvm_ir.clone(),
                spirv_binary,
                stage: checkpoint.compilation_stage.clone(),
                debug_mappings: checkpoint.debug_mappings.clone(),
                compile_options: checkpoint.compile_options.clone(),
            })
        } else {
            Err(CheckpointError::CheckpointNotFound(
                checkpoint_id.to_string(),
            ))
        }
    }

    /// 生成检查点报告
    pub fn generate_report(&self) -> CheckpointReport {
        let total_checkpoints = self.active_checkpoints.len();
        let mut stage_counts = HashMap::new();
        let mut total_compile_time = 0u64;
        let mut failed_count = 0;

        for checkpoint in self.active_checkpoints.values() {
            *stage_counts
                .entry(checkpoint.compilation_stage.clone())
                .or_insert(0) += 1;
            total_compile_time += checkpoint.performance_stats.total_time_ms;

            if checkpoint.compilation_stage == CompilationStage::Failed {
                failed_count += 1;
            }
        }

        CheckpointReport {
            total_checkpoints,
            stage_distribution: stage_counts,
            average_compile_time_ms: if total_checkpoints > 0 {
                total_compile_time / total_checkpoints as u64
            } else {
                0
            },
            success_rate: if total_checkpoints > 0 {
                ((total_checkpoints - failed_count) as f64 / total_checkpoints as f64) * 100.0
            } else {
                0.0
            },
            disk_usage_bytes: self.calculate_disk_usage(),
        }
    }

    /// 计算磁盘使用量
    fn calculate_disk_usage(&self) -> u64 {
        let mut total_size = 0u64;

        if let Ok(entries) = fs::read_dir(&self.checkpoint_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    total_size += metadata.len();
                }
            }
        }

        total_size
    }
}

/// 编译状态恢复信息
#[derive(Debug, Clone)]
pub struct CompilationRestoreInfo {
    pub ptx_source: String,
    pub llvm_ir: Option<String>,
    pub spirv_binary: Option<Vec<u8>>,
    pub stage: CompilationStage,
    pub debug_mappings: Vec<DwarfMappingEntry>,
    pub compile_options: CompileOptions,
}

/// 检查点报告
#[derive(Debug)]
pub struct CheckpointReport {
    pub total_checkpoints: usize,
    pub stage_distribution: HashMap<CompilationStage, usize>,
    pub average_compile_time_ms: u64,
    pub success_rate: f64,
    pub disk_usage_bytes: u64,
}

/// 检查点错误类型
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("Checkpoint not found: {0}")]
    CheckpointNotFound(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Invalid checkpoint data: {0}")]
    InvalidData(String),
}

/// 便利的宏用于创建检查点
#[macro_export]
macro_rules! create_checkpoint {
    ($manager:expr, $ptx:expr, $stage:expr, $desc:expr) => {
        $manager.create_checkpoint($ptx.to_string(), $stage, $desc.to_string())
    };
}

/// 便利的宏用于更新检查点
#[macro_export]
macro_rules! update_checkpoint {
    ($manager:expr, $id:expr, $stage:expr) => {
        $manager.update_checkpoint($id, $stage)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let ptx_source = ".version 7.0\n.target sm_50\n.kernel test() { ret; }";
        let id = manager.create_checkpoint(
            ptx_source.to_string(),
            CompilationStage::PtxParsing,
            "测试检查点".to_string(),
        );

        assert!(manager.get_checkpoint(&id).is_some());
        let checkpoint = manager.get_checkpoint(&id).unwrap();
        assert_eq!(checkpoint.ptx_source, ptx_source);
        assert_eq!(checkpoint.compilation_stage, CompilationStage::PtxParsing);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let ptx_source = ".version 7.0\n.target sm_50\n.kernel test() { ret; }";
        let id = manager.create_checkpoint(
            ptx_source.to_string(),
            CompilationStage::PtxParsing,
            "保存测试".to_string(),
        );

        // 保存检查点
        let filepath = manager.save_checkpoint(&id).unwrap();
        assert!(filepath.exists());

        // 创建新的管理器并加载检查点
        let mut new_manager = CheckpointManager::new(temp_dir.path()).unwrap();
        let loaded_id = new_manager.load_checkpoint(&filepath).unwrap();

        assert_eq!(id, loaded_id);
        let loaded_checkpoint = new_manager.get_checkpoint(&loaded_id).unwrap();
        assert_eq!(loaded_checkpoint.ptx_source, ptx_source);
    }

    #[test]
    fn test_checkpoint_update() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let id = manager.create_checkpoint(
            "test".to_string(),
            CompilationStage::PtxParsing,
            "更新测试".to_string(),
        );

        // 更新阶段
        manager
            .update_checkpoint(&id, CompilationStage::LlvmGeneration)
            .unwrap();

        let checkpoint = manager.get_checkpoint(&id).unwrap();
        assert_eq!(
            checkpoint.compilation_stage,
            CompilationStage::LlvmGeneration
        );
    }
}
