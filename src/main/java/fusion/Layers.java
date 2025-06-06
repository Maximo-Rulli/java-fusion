package fusion;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.lang.Math;

// Basic operations used in the U-Net architecture
public class Layers {
  // Weights W, Biases b
  private INDArray W;
  private INDArray b;
  
  public Layers(String W_name, String b_name){
    //Load from .npy files stored in /weights
    W = Nd4j.readNpy(new File("weights/" + W_name + "_weight.npy"));
    b = Nd4j.readNpy(new File("weights/" + b_name + "_bias.npy"));
  }

  public Layers(INDArray W, INDArray b){
    //Set the weights to the provided matrices (debug testing only)
    this.W = W;
    this.b = b;
  }

  public INDArray Conv(INDArray input, int stride, int padding) {
    final int outChannels = (int) this.W.shape()[0];
    final int kernelSize = (int) this.W.shape()[3];
    
    // We transform the input into intermediate representation to then flatten it
    INDArray patches = Convolution.im2col(input, kernelSize, kernelSize, stride, stride, padding, padding, false);
    //Output shape: [N, Channels, Kernel_H, Kernel_W, Out_H, Out_W] -- For our use-case N=1 always

    //System.out.println(patches);
    
    // Reshape patches to be multiplied by single vector of kernel
    INDArray colReshaped = patches.permute(0, 4, 5, 1, 2, 3)  // [1, Out_H, Out_W, Channels, Kernel_H, Kernel_W]
                            .reshape('c', 
                            new long[]{
                              1 * patches.shape()[4] * patches.shape()[5], 
                              patches.shape()[1] * patches.shape()[2] * patches.shape()[3]
                            });
    //Output shape: [N * Out_H * Out_W, Channels * Kernel_H * Kernel_W]

    // Reshape kernel to a vector to apply it to reshaped patches
    INDArray kernelReshaped = this.W.reshape('c', 
                            new long[]{
                              W.shape()[0], 
                              W.shape()[1] * W.shape()[2] * W.shape()[3]
                            });
    //Output shape: [outChannels, inChannels * Kernel_H * Kernel_W]

    // Multiply colReshaped with kernel.T, equivalent to applying the filters correspondingly
    INDArray result = colReshaped.mmul(kernelReshaped.transpose());
    //Output shape: [N * Out_H * Out_W, outChannels]
    
    // Reshape to [N, outChannels, Out_H, Out_W]
    INDArray out = result.reshape('c', 1, outChannels, patches.shape()[4], patches.shape()[5]);
    
    System.out.println(Arrays.toString(out.shape()));
    //System.out.println(out);
    return out;
  }

  public static INDArray maxPool(INDArray input, int kernelSize, int stride) {
    // Extract shape from input (# channels/layers, height, width)
    // It's supposed to use concurrency, so each image calls a different function
    long[] shape = input.shape();

    // Assume that input is always square, so output width = output height
    int out_shape =  Math.floorDiv(((int) shape[2]-kernelSize), stride)+1;

    // Create empty output with corresponding shape
    INDArray out = Nd4j.zeros(shape[0], out_shape, out_shape);

    // Main for loop where the Max-pooling is done

    // Iteration over channels
    for (int c=0; c<shape[0]; c++){
      // Iteration inside a channel
      for (int i=0; i*stride+kernelSize-1<shape[1]; i++){
        for (int j=0; j*stride+kernelSize-1<shape[2]; j++){
  
          // Slice array part that the pooling will be applied at
          INDArray slice = input.get(
            NDArrayIndex.point(c),
            NDArrayIndex.interval(i*stride, i*stride+kernelSize),
            NDArrayIndex.interval(j*stride, j*stride+kernelSize)
          );
          
          // Extract maxNumber of slice and put it in corresponding output position
          out.putScalar(new int[] {c,i,j}, slice.maxNumber().floatValue());
        }       
      }
    }
    
    //System.out.println(out);
    return out;
  }

  public static INDArray maxPool(INDArray input){
    return maxPool(input, 2, 1);
  }

  public static INDArray TranspConv(INDArray input, int inChannels, int outChannels, int kernelSize, int stride, int out_padding) {
    return input;
  }

  public static INDArray concat(INDArray x1, INDArray x2) {
    return Nd4j.concat(0, x1, x2);
  }
}
