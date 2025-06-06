package fusion;

import java.util.List;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.factory.Nd4j;

public class Correctness {
  public static void run() {
    ConvTest.run();
  }

  private static class ConvTest {
    private static void run(){

      final int padding = 1;
      final int stride = 1;

      List<INDArray> outMats = StandardMatrices.UpChannelsMat();

      SDVariable convOut = config(outMats.get(0), outMats.get(1), padding, stride);

      INDArray output = convOut.eval();
      
      System.out.println(output);
    }
  }

  private static SDVariable config(INDArray input, INDArray kernel, int padding, int stride) {
    final int kernelSize = (int) kernel.shape()[0];

    SameDiff sd = SameDiff.create();
    SDVariable x = sd.constant("x", input);
    SDVariable w = sd.constant("w", kernel);

    Conv2DConfig config = Conv2DConfig.builder()
        .kH(kernelSize).kW(kernelSize)
        .sH(stride).sW(stride)
        .pH(padding).pW(padding)
        .dataFormat("NCHW")
        .build();

    SDVariable convolution = sd.cnn().conv2d(x, w, config);

    return convolution;
  }
}
