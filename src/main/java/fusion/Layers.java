package fusion;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// Basic operations used in the U-Net architecture
public class Layers {
  public static INDArray Conv(INDArray input, int inChannels, int outChannels, int kernelSize, int stride, int padding) {
    return input;
  }

  public static INDArray maxPool(INDArray input, int kernelSize, int stride) {
    System.out.println();
    
    return input;
  }

  public static INDArray TranspConv(INDArray input, int inChannels, int outChannels, int kernelSize, int stride, int out_padding) {
    return input;
  }

  public static INDArray concat(INDArray x1, INDArray x2) {
    return x1;
  }
}
