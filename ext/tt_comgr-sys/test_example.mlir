// Test MLIR file for TTMLIR processing
// This simulates test/ttmlir/EmitC/TTNN/sanity_add.mlir

module {
  func.func @sanity_add(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}