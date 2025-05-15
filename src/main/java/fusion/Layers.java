package fusion;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.lang.Math;

// Basic operations used in the U-Net architecture
public class Layers {
  public static INDArray Conv(INDArray input, int inChannels, int outChannels, int kernelSize, int stride, int padding) {
    return input;
  }

  public static INDArray maxPool(INDArray input, int kernelSize, int stride) {
    // Extract shape from input (# images, # channels/layers, height, width)
    long[] shape = input.shape();

    // Assume that input is always square, so output width = output height
    int out_shape =  Math.floorDiv(((int) shape[3]-kernelSize), stride)+1;

    // Create empty output with corresponding shape
    INDArray out = Nd4j.zeros(shape[0], shape[1], out_shape, out_shape);

    // Main for loop where the Max-pooling is done
    for (int i=0; i*stride+kernelSize-1<shape[2]; i++){
      for (int j=0; j*stride+kernelSize-1<shape[3]; j++){

        // Slice array part that the pooling will be applied at
        INDArray slice = input.get(
          NDArrayIndex.all(),
          NDArrayIndex.all(),
          NDArrayIndex.interval(i*stride, i*stride+kernelSize),
          NDArrayIndex.interval(j*stride, j*stride+kernelSize)
        );
        
        // Extract maxNumber of slice and put it in corresponding output position
        out.putScalar(new int[] {0,0,i,j}, slice.maxNumber().floatValue());
      }       
    }
    
    System.out.println(out);
    return out;
  }

  public static INDArray maxPool(INDArray input){
    return maxPool(input, 2, 1);
  }

  public static INDArray TranspConv(INDArray input, int inChannels, int outChannels, int kernelSize, int stride, int out_padding) {
    return input;
  }

  public static INDArray concat(INDArray x1, INDArray x2) {
    return x1;
  }
}
