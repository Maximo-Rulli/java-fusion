package fusion;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.lang.Math;

// Basic operations used in the U-Net architecture
public class Layer {
  // Weights W, Biases b
  private INDArray W;
  private INDArray b;
  
  public Layer(String W_name, String b_name){
    //Load from .npy files stored in /weights
    W = Nd4j.readNpy(new File("weights/" + W_name + "_weight.npy"));
    b = Nd4j.readNpy(new File("weights/" + b_name + "_bias.npy"));
  }

  public Layer(INDArray W, INDArray b){
    //Set the weights to the provided matrices (debug testing only)
    this.W = W;
    this.b = b;
  }

  public INDArray Conv(INDArray input, int stride, int padding) {
    return Conv(input, stride, padding, true, false);
  }

  public INDArray Conv(INDArray input, int stride, int padding, boolean bias, boolean debug) {
    final int outChannels = (int) this.W.shape()[0];
    final int kernelSize = (int) this.W.shape()[3];

    // We transform the input into intermediate representation to then flatten it
    INDArray patches = Convolution.im2col(input, kernelSize, kernelSize, stride, stride, padding, padding, false);
    //Output shape: [N, Channels, Kernel_H, Kernel_W, Out_W, Out_H] -- For our use-case N=1 always
    
    // Reshape patches to be multiplied by single vector of kernel
    INDArray colReshaped = patches.permute(0, 4, 5, 1, 2, 3)  // [1, Out_H, Out_W, Channels, Kernel_H, Kernel_W]
                            .reshape('c', 
                            new long[]{
                              patches.shape()[4] * patches.shape()[5], 
                              patches.shape()[1] * patches.shape()[2] * patches.shape()[3]
                            });
    //Output shape: [N * Out_H * Out_W, Channels * Kernel_H * Kernel_W]

    // Reshape kernel to a vector to apply it to reshaped patches
    INDArray kernelReshaped = this.W.reshape('c', 
                            new long[]{
                              outChannels, 
                              W.shape()[1] * kernelSize * kernelSize
                            });
    //Output shape: [outChannels, inChannels * Kernel_H * Kernel_W]

    // Multiply colReshaped with kernel.T, equivalent to applying the filters correspondingly
    INDArray result = colReshaped.mmul(kernelReshaped.transpose());
    //Output shape: [N * Out_H * Out_W, outChannels]
    
    // Reshape to [N, outChannels, Out_H, Out_W]
    INDArray out = result.transpose()
                            .reshape(result.shape()[0]*result.shape()[1]) // First we flatten the obtained array
                            .reshape(1,outChannels,patches.shape()[4],patches.shape()[5]); //Then reshape (only way to make it work)

    if (bias){
      INDArray summableb = (this.b).reshape(1,outChannels,1,1).broadcast(1,outChannels,patches.shape()[4],patches.shape()[5]);
      out = out.add(summableb);
    }

    if (debug){
      System.out.println("Input:\n"+input);
      System.out.println("Patches:\n"+patches);
      System.out.println("Colreshaped:\n"+colReshaped);
      System.out.println("Kernel:\n"+this.W);
      System.out.println("Reshaped Kernel (after transpose):\n"+kernelReshaped.transpose());
      System.out.println("Output:\n"+out);
    }

    return out;
  }

  public INDArray TranspConv(INDArray input, int stride, int padding, boolean debug) {
    // Extract input and kernel shape info
    final int C = (int) input.shape()[1];
    final int H = (int) input.shape()[2];
    final int W = (int) input.shape()[3];

    int outC = (int) this.W.shape()[0];
    int kernelSize = (int) this.W.shape()[2];

    // 1. Reshape input to [Channels, H * W]
    INDArray inputFlat = input.reshape(C, H * W); // [Channels, H*W]

    // 2. Reshape kernel to [outChannels, Channels * kernelH * kernelW]
    INDArray kernelFlat = this.W.reshape(outC, -1);

    // 3. Multiply: [H*W, Channels] x [Channels, outChannels * kernelH * kernelW]
    INDArray patches = inputFlat.transpose().mmul(kernelFlat); // [H*W, outChannels * kH * kW]

    // 4. Reshape to col2im format: [1, outChannels, H, W, kernelH, kernelW]
    INDArray patchesReshaped = patches.transpose()                        // [Channels * kernelH * kernelW, H*W]
            .reshape(outC, kernelSize, kernelSize, H, W)                              // [Channels, kernelH, kernelW, H, W]
            .permute(0, 3, 4, 1, 2)                                       // [Channels, H, W, kernelH, kernelW]
            .reshape(1, outC, H, W, kernelSize, kernelSize);                          // [1, Channels, H, W, kernelH, kernelW]

    // 5. Output shape before cropping
    int fullOutH = (H - 1) * stride + kernelSize;
    int fullOutW = (W - 1) * stride + kernelSize;

    // 6. Use col2im to scatter patches into output image
    INDArray output = Convolution.col2im(
      patchesReshaped,
      stride, stride,
      0, 0, // we already handled padding by target shape
      fullOutH,
      fullOutW
    );

    // 7. Crop output to simulate padding (transpose conv "pads" output)
    if (padding > 0) {
      output = output.get(
        NDArrayIndex.all(),
        NDArrayIndex.all(),
        NDArrayIndex.interval(padding, padding + (fullOutH - 2 * padding)),
        NDArrayIndex.interval(padding, padding + (fullOutW - 2 * padding))
      );
    }

      return output;
  }

  public static INDArray concat(INDArray x1, INDArray x2) {
    return Nd4j.concat(0, x1, x2);
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

}
