package fusion;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// U-Net-like neural network (architecture only)
public class UNet {
  private INDArray W;
  private INDArray b;

  public UNet(String path) {
    Map<String, INDArray> weights = new HashMap<>();
    Map<String, INDArray> biases = new HashMap<>();
    INDArray W_enc1_0 = Nd4j.readNpy(new File(path+"enc1_0_weight.npy"));
    INDArray b_enc1_0 = Nd4j.readNpy(new File(path+"enc1_0_bias.npy"));
  }

  public double[] predict(double[] x, int t) {
    return x;
  }

  private double[] convBlock(double[] x, int inChannels, int outChannels) {
    return x;
  }

  private double[] upSample(double[] x, int inChannels, int outChannels) {
    return x;
  }

  private double[] concat(double[] x1, double[] x2) {
    return x1;
  }
}
