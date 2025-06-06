package fusion;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
//import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4j;

public class Correctness{
  enum convType {
    UP,
    SAME,
    DOWN,
  }

  public static void run() throws IOException, InterruptedException{
    ConvTest.run();
  }

  private static class ConvTest {
    private static void run() throws IOException, InterruptedException{

      final int padding = 0;
      final int stride = 1;

      List<INDArray> outMats = StandardMatrices.UpChannelsMat(false);
      Layer UpChannelLayer = new Layer(outMats.get(1), outMats.get(1));
      INDArray layerOut = UpChannelLayer.Conv(outMats.get(0), stride, padding);
      INDArray UpChannelMatOut = Nd4j.readBinary(new File("src/main/java/fusion/correct_outs/UpChannelMatOut.bin"));

      if (!UpChannelMatOut.equalsWithEps(layerOut, 1e-5)){
        System.out.println("\nError while assessing convolution correctness, incorrect output returned.\n");
        System.out.println("Returned matrix (wrong):\n" +  layerOut);
        System.out.println("\nExpected output:\n" + UpChannelMatOut);
        throw new AssertionError("Convolution test failed.");
      }
      
      outMats = StandardMatrices.SameChannelsMat(false);
      Layer SameChannelLayer = new Layer(outMats.get(1), outMats.get(1));

      outMats = StandardMatrices.DownChannelsMat(false);
      Layer DownChannelLayer = new Layer(outMats.get(1), outMats.get(1));

      
      //generateSample(padding, stride, convType.DOWN);
      
      //Utils.printld("Verificando convolución", "Convolución verificada con éxito!");
    }
  }

  private static void generateSample(int padding, int stride, convType convolutionType) throws IOException{
    List<INDArray> outMats;

    if (convolutionType == convType.DOWN)
      outMats = StandardMatrices.SameChannelsMat(null);
    
    else if (convolutionType == convType.DOWN)
      outMats = StandardMatrices.SameChannelsMat(null);
    
    else
      outMats = StandardMatrices.SameChannelsMat(null);
    
    SDVariable convOut = config(outMats.get(0), outMats.get(1), padding, stride);

    INDArray output = convOut.eval();
    Nd4j.saveBinary(output, new File("DownChannelMatOutp"+ padding + ".bin"));
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
