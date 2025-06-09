package fusion;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.factory.Nd4j;

public class Correctness{
  enum convType {
    UP,
    SAME,
    DOWN,
    TRANSP,
    MODEL
  }
  UNet Model;

  public Correctness(UNet Model){
    this.Model = Model;
  }

  public void run() throws IOException, InterruptedException{
    ConvTest.run();
    TranspConvTest.run();
    ModelTest MTest = new ModelTest();
    MTest.run();
  }

  private static class ConvTest {
    private static void run() throws IOException, InterruptedException{
      int stride = 1;
      
      int padding = 0;
      List<INDArray> outMats = StandardMatrices.UpChannelsMat(false);
      Layer ConvLayer = new Layer(outMats.get(1), outMats.get(1));
      INDArray layerOut = ConvLayer.Conv(outMats.get(0), stride, padding, false, false);
      INDArray TrueOut = Nd4j.readBinary(new File("src/main/java/fusion/correct_outs/conv/UpChannelMatOutp0.bin"));
      checkError(layerOut, TrueOut, convType.UP);


      padding = 1;
      outMats = StandardMatrices.SameChannelsMat(null, null, false);
      ConvLayer = new Layer(outMats.get(1), outMats.get(1));
      layerOut = ConvLayer.Conv(outMats.get(0), stride, padding, false, false);
      TrueOut = Nd4j.readBinary(new File("src/main/java/fusion/correct_outs/conv/SameChannelMatOutp1.bin"));
      checkError(layerOut, TrueOut, convType.SAME);

      
      padding = 0;
      outMats = StandardMatrices.DownChannelsMat(false);
      ConvLayer = new Layer(outMats.get(1), outMats.get(1));
      layerOut = ConvLayer.Conv(outMats.get(0), stride, padding, false, false);
      TrueOut = Nd4j.readBinary(new File("src/main/java/fusion/correct_outs/conv/DownChannelMatOutp0.bin"));
      checkError(layerOut, TrueOut, convType.DOWN);

      
      stride = 2;
      padding = 1;
      // 128->256 channels convolution on 16x16 input
      INDArray Weights = Nd4j.readNpy(new File("src/main/java/fusion/correct_outs/conv/Conv16x16s2p1ic128oc256-rand-w.npy"));
      INDArray Input = Nd4j.readNpy(new File("src/main/java/fusion/correct_outs/conv/Conv16x16s2p1ic128oc256-rand-in.npy"));
      ConvLayer = new Layer(Weights, Weights);
      layerOut = ConvLayer.Conv(Input, stride, padding, false, false);
      TrueOut = Nd4j.readNpy(new File("src/main/java/fusion/correct_outs/conv/Conv16x16s2p1ic128oc256-rand-out.npy"));
      checkError(layerOut, TrueOut, convType.UP);
      
      //generateSample(0, 1, convType.UP, null, null);
      
      Utils.printld("Verificando convolución", "Convolución verificada con éxito!");
    }
  }


  private static class TranspConvTest {
    private static void run() throws IOException, InterruptedException{

      int stride = 1;
      int padding = 0;
      List<INDArray> outMats = StandardMatrices.TranspMat(3, 3, 4);
      Layer TranspLayer = new Layer(outMats.get(1), outMats.get(1));
      INDArray layerOut = TranspLayer.TranspConv(outMats.get(0), stride, padding, false);
      INDArray TrueOut = Nd4j.readBinary(new File("src/main/java/fusion/correct_outs/transp/TranspConv3x3s1p0.bin"));
      checkError(layerOut, TrueOut, convType.TRANSP);

      stride = 1;
      padding = 1;
      outMats = StandardMatrices.TranspMat(5, 5, 3);
      TranspLayer = new Layer(outMats.get(1), outMats.get(1));
      layerOut = TranspLayer.TranspConv(outMats.get(0), stride, padding, false);
      TrueOut = Nd4j.readBinary(new File("src/main/java/fusion/correct_outs/transp/TranspConv5x5s1p1.bin"));
      checkError(layerOut, TrueOut, convType.TRANSP);
      
      stride = 2;
      padding = 1;
      outMats = StandardMatrices.TranspMat(5, 5, 3);
      TranspLayer = new Layer(outMats.get(1), outMats.get(1));
      layerOut = TranspLayer.TranspConv(outMats.get(0), stride, padding, false);
      TrueOut = Nd4j.readBinary(new File("src/main/java/fusion/correct_outs/transp/TranspConv5x5s2p1.bin"));
      checkError(layerOut, TrueOut, convType.TRANSP);
      
      
      stride = 2;
      padding = 1;
      // 128 channels transpose convolution on 10x10 input
      INDArray Weights = Nd4j.readNpy(new File("src/main/java/fusion/correct_outs/transp/TranspConv10x10s2p1c128-rand-w.npy"));
      INDArray Input = Nd4j.readNpy(new File("src/main/java/fusion/correct_outs/transp/TranspConv10x10s2p1c128-rand-in.npy"));
      TranspLayer = new Layer(Weights, Weights);
      layerOut = TranspLayer.TranspConv(Input, stride, padding, false);
      TrueOut = Nd4j.readNpy(new File("src/main/java/fusion/correct_outs/transp/TranspConv10x10s2p1c128-rand-out.npy"));
      checkError(layerOut, TrueOut, convType.TRANSP);
      
      Utils.printld("Verificando convolución transpuesta", "Convolución transpuesta verificada con éxito!");
    }
  }
  
  private class ModelTest {
    private  void run() throws IOException, InterruptedException{
      INDArray sampleInput = (Nd4j.create(new float[]{2})).broadcast(1,1,28,28);
      INDArray sampleTime = Nd4j.create(new float[]{1}, new int[]{1});
      INDArray predicted = Correctness.this.Model.predict(sampleInput, sampleTime);
      INDArray TrueOut = Nd4j.readBinary(new File("src/main/java/fusion/correct_outs/java_fusion.bin"));
      checkError(predicted, TrueOut, convType.MODEL);
      Utils.printld("Verificando modelo completo", "Modelo verificado con éxito!");
    }
  }
  
  private static void generateSample(int padding, int stride, convType convolutionType, Integer Height, Integer Width) throws IOException{
    List<INDArray> outMats;
    String filename;

    if (convolutionType == convType.DOWN){
      outMats = StandardMatrices.DownChannelsMat(null);
      filename = "DownChannelMatOutp";
    }
    else if (convolutionType == convType.SAME){
      outMats = StandardMatrices.SameChannelsMat(Height, Width, null);
      filename = "SameChannelMatOutp";
    }
    else {
      outMats = StandardMatrices.UpChannelsMat(null);
      filename = "UpChannelMatOutp";
    }

    SDVariable convOut;
    INDArray output;

    convOut = config(outMats.get(0), outMats.get(1), padding, stride);
    output = convOut.eval();
    Nd4j.saveBinary(output, new File(filename + padding + ".bin"));

  }


  private static void checkError(INDArray layerOut, INDArray TrueOut, convType Type){
    if (!TrueOut.equalsWithEps(layerOut, 1e-3)){
      if (Type == convType.TRANSP){
        System.out.println("\nError while assessing transpose convolution correctness, incorrect output returned.\n");
      }
      else if  (Type != convType.MODEL) {
        System.out.println("\nError while assessing convolution correctness, incorrect output returned.\n");
      }
      else {
        System.out.println("\nError while assessing model correctness, incorrect output returned.\n");
      }
      System.out.println("Shape: "+Arrays.toString(TrueOut.shape()));
      System.out.println("Returned matrix (wrong):\n" +  layerOut);
      System.out.println("\nShape: "+Arrays.toString(TrueOut.shape()));
      System.out.println("Expected output:\n" + TrueOut);
      throw new AssertionError("Correctness test failed.");
    }
  }

  
  private static SDVariable config(INDArray input, INDArray kernel, int padding, int stride) {
    final int kernelSize = (int) kernel.shape()[2];

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