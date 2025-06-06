// This file contains standard matrices that are used alongside Correctness to check correctness of implemented functions

package fusion;

import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class StandardMatrices {
  static int H = 5;
  static int W = 5;
  static int kernelSize = 3;

  // Matrices representing an increase of output channels
  public static List<INDArray> UpChannelsMat() {
    final int C = 2;
    final int outChannels = 3;

    return builder(C, outChannels);
  }

  // Matrices representing a preservation of output channels
  public static List<INDArray> SameChannelsMat() {
    final int C = 3;
    final int outChannels = 3;

    return builder(C, outChannels);
  }

  // Matrices representing a decrease of output channels
  public static List<INDArray> DownChannelsMat() {
    final int C = 3;
    final int outChannels = 2;

    return builder(C, outChannels);
  }

  private static List<INDArray> builder(int C, int outChannels){
    // Input: [N, C, H, W]
    INDArray input = Nd4j.rand(new int[]{1, C, H, W});

    // Kernel: [kernelH, kernelW, inputChannels, outputChannels]
    INDArray kernel = Nd4j.rand(new int[]{kernelSize, kernelSize, C, outChannels});

    List<INDArray> builtMats = Arrays.asList(input, kernel);

    return builtMats;
  }
}
