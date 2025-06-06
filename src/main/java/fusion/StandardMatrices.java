// This file contains standard matrices that are used alongside Correctness to check correctness of implemented functions

package fusion;

import java.util.Arrays;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class StandardMatrices {
  static int H = 5;
  static int W = 5;
  static int kernelSize = 3;

  // Matrices representing an increase of output channels
  public static List<INDArray> UpChannelsMat(Boolean tfStruct) {
    final int C = 2;
    final int outChannels = 3;

    return (tfStruct == null ? builder(C, outChannels) : builder(C, outChannels, tfStruct));
  }

  // Matrices representing a preservation of output channels
  public static List<INDArray> SameChannelsMat(Boolean tfStruct) {
    final int C = 3;
    final int outChannels = 3;

    return (tfStruct == null ? builder(C, outChannels) : builder(C, outChannels, tfStruct));
  }

  // Matrices representing a decrease of output channels
  public static List<INDArray> DownChannelsMat(Boolean tfStruct) {
    final int C = 3;
    final int outChannels = 2;

    return (tfStruct == null ? builder(C, outChannels) : builder(C, outChannels, tfStruct));
  }

  private static List<INDArray> builder(int C, int outChannels){
    return builder(C, outChannels, true);
  }

  private static List<INDArray> builder(int C, int outChannels, boolean tfStruct){
    //tfStruct is whether we are using tensorflows structure of building kernels
    //or pytorch's. Tf:[kernelH, kernelW, input, output] Torch:[output, input, kernelW, kernelH]
    //Note that all the loaded weights will be in Torch format

    // Input shape: [N, C, H, W]
    INDArray input = Nd4j.create(new int[]{1, C, H, W});
    
    // Assign input[c, h, w] = c + h + w for variety
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input.putScalar(new int[]{0, c, h, w}, c + h + w);
            }
        }
    }

    INDArray kernel;

    if (tfStruct) {
      // Kernel shape: [kernelH, kernelW, input, output]
      kernel = Nd4j.create(new int[]{kernelSize, kernelSize, C, outChannels});
      
      // Each (:, :, c, f) filter is filled with (f + 1) * (c + 1)
      for (int f = 0; f < outChannels; f++) {
          for (int c = 0; c < C; c++) {
              kernel.get(all(), all(), point(c), point(f)).assign((f + 1) * (c + 1));
          }
      }
    }
    else {
      // Kernel shape: [output, input, kernelW, kernelH]
      kernel = Nd4j.create(new int[]{outChannels, C, kernelSize, kernelSize});
      
      // Each (:, :, c, f) filter is filled with (f + 1) * (c + 1)
      for (int f = 0; f < outChannels; f++) {
          for (int c = 0; c < C; c++) {
              kernel.get(point(f), point(c), all(), all()).assign((f + 1) * (c + 1));
          }
      }
    }

    return Arrays.asList(input, kernel);
  }
}
