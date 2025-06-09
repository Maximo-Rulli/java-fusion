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

    return (tfStruct == null ? builder(C, outChannels, null, null) : builder(C, outChannels, kernelSize, null, null, tfStruct));
  }

  // Matrices representing a preservation of output channels
  public static List<INDArray> SameChannelsMat(Integer Height, Integer Width, Boolean tfStruct) {
    final int C = 3;
    final int outChannels = 3;

    return (tfStruct == null ? builder(C, outChannels, Height, Width) : builder(C, outChannels, kernelSize, Height, Width, tfStruct));
  }

  // Matrices representing a decrease of output channels
  public static List<INDArray> DownChannelsMat(Boolean tfStruct) {
    final int C = 3;
    final int outChannels = 2;

    return (tfStruct == null ? builder(C, outChannels, null, null) : builder(C, outChannels, kernelSize, null, null, tfStruct));
  }

  // Special matrices for transpose convolution (similar logic to other ones)
  public static List<INDArray> TranspMat(Integer Height, Integer Width, int C){
    final int kernel = 4;
    // By the design of our U-Net the input and output channels always have = dimension on transpose convolution
    return builder(C, C, kernel, Height, Width, false);
  }

  private static List<INDArray> builder(int C, int outChannels, Integer Height, Integer Width){
    return builder(C, outChannels, Height, kernelSize, Width, true);
  }

  private static List<INDArray> builder(int C, int outChannels, int kernelS, Integer Height, Integer Width, boolean tfStruct){
    //tfStruct is whether we are using tensorflows structure of building kernels
    //or pytorch's. Tf:[kernelH, kernelW, input, output] Torch:[output, input, kernelW, kernelH]
    //Note that all the loaded weights will be in Torch format
    int inH, inW;
    if (Height == null && Width == null){
      inH = H; inW = W;
    }
    else {
      inH = Height; inW = Width;
    }

    // Input shape: [N, C, H, W]
    INDArray input = Nd4j.create(new int[]{1, C, inH, inW});
    
    // Assign input[c, h, w] = c + h + w for variety
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < inH; h++) {
            for (int w = 0; w < inW; w++) {
                input.putScalar(new int[]{0, c, h, w}, c + h + w);
            }
        }
    }

    INDArray kernel;

    if (tfStruct) {
      // Kernel shape: [kernelH, kernelW, input, output]
      kernel = Nd4j.create(new int[]{kernelS, kernelS, C, outChannels});
      
      // Each (:, :, c, f) filter is filled with (f + 1) * (c + 1)
      for (int f = 0; f < outChannels; f++) {
        for (int c = 0; c < C; c++) {
          kernel.get(all(), all(), point(c), point(f)).assign((f + 1) * (c + 1));
        }
      }
    }
    else {
      // Kernel shape: [output, input, kernelW, kernelH]
      kernel = Nd4j.create(new int[]{outChannels, C, kernelS, kernelS});
      
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
