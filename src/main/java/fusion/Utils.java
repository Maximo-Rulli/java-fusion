package fusion;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
// Utility class for image operations and noise handling
public class Utils {
    public static INDArray timestepEmbedding(INDArray timesteps) {
        return timestepEmbedding(timesteps, 64, 10000);
    }

    public static INDArray timestepEmbedding(INDArray timesteps, int dim, int maxPeriod) {
      int half = dim / 2;

      // Create [0, 1, ..., half-1]
      INDArray arange = Nd4j.arange(half);

      // Compute the frequencies: exp(-log(maxPeriod) * i / half)
      double logMaxPeriod = Math.log(maxPeriod);
      INDArray freqs = Transforms.exp(arange.mul(-logMaxPeriod / half));

      // Reshape timesteps: [batchSize, 1]
      INDArray timestepsFloat = timesteps.castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT);
      timestepsFloat = timestepsFloat.reshape(timesteps.length(), 1);

      // Multiply timesteps by frequencies -> [batchSize, half]
      INDArray args = timestepsFloat.mmul(freqs.reshape(1, half));

      INDArray cosPart = Transforms.cos(args);
      INDArray sinPart = Transforms.sin(args);

      // Concatenate cos and sin parts along last dimension
      INDArray embedding = Nd4j.concat(1, cosPart, sinPart);

      return embedding;
    }

    public static void saveImage(double[] image, String filename) {}

    public static void printld(String loadText, String endText) throws InterruptedException{
      System.out.print("\u001B[?25l");
      int maxDots = 4;

      for (int i = 0; i <= 30; i++) {
          int dotCount = i % maxDots;
          String dots = ".".repeat(dotCount);

          System.out.print("\r" + loadText + dots);

          // Erase leftover dots if we just went from more dots to fewer dots
          if (dotCount < maxDots) {
              // overwrite extra dots with spaces
              System.out.print(" ".repeat(maxDots - dotCount));
          }

          Thread.sleep(250);
      }
      System.out.println();
      System.out.println(endText);
      System.out.print("\u001B[?25h");
    }
}