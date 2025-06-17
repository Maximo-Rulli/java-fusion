package fusion;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.lang.Math;

// Basic operations used in the U-Net architecture
public class Layer {
  // Weights W, Biases b
  public INDArray W;
  public INDArray b;

  enum layerType {
    Conv,
    TranspConv,
    Linear
  }
  
  public Layer(String path, layerType lType){
    //Load from .npy files stored in /weights
    W = Nd4j.readNpy(new File("weights/" + path + "_weight.npy"));
    b = Nd4j.readNpy(new File("weights/" + path + "_bias.npy"));

    if (lType == layerType.TranspConv){
      /// This is because in Transpose convolution of torch, dimensions are swapped (cOut, cIn, H, W)
      /// https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
      this.W = this.W.permute(1,0,2,3);
    }
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
    final int outChannels = (int) W.shape()[0];
    final int kernelSize = (int) W.shape()[3];

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
    INDArray kernelReshaped = W.reshape('c', 
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

  /*public INDArray TranspConv(INDArray input, int stride, int padding) {
    // Dilate input (insert zeros between elements for stride > 1)
    INDArray dilatedInput = (stride > 1) ? dilateInput(input, stride) : input;
    
    // 2. Calculate effective padding
    int kernelH = (int) this.W.shape()[2];
    int effectivePadding = kernelH - 1 - padding;
    
    // Pad the dilated input
    INDArray paddedInput = Nd4j.pad(dilatedInput, 
                            new int[][]{{0,0}, {0,0}, 
                              {effectivePadding, effectivePadding}, 
                              {effectivePadding, effectivePadding}} 
                            );
    
    // 4. Flip kernel (180-degree rotation)
   // INDArray flippedKernel = flipKernel(this.W);
    
    // Regular convolution with stride=1
    return this.Conv(paddedInput, 1, 0, false, false);
  }

  private INDArray dilateInput(INDArray input, int stride) {
    // Get input dimensions [batch, channels, height, width]
    long[] inputShape = input.shape();
    long batch = inputShape[0];
    long channels = inputShape[1];
    long height = inputShape[2];
    long width = inputShape[3];
    
    // Calculate dilated dimensions
    long dilatedHeight = height + (height - 1) * (stride - 1);
    long dilatedWidth = width + (width - 1) * (stride - 1);
    
    // Create output array filled with zeros
    INDArray dilated = Nd4j.zeros(batch, channels, dilatedHeight, dilatedWidth);
    
    // Fill the dilated array by placing original values at strided positions
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // Place original value at dilated position
                    int dilatedH = h * stride;
                    int dilatedW = w * stride;
                    
                    double value = input.getDouble(b, c, h, w);
                    dilated.putScalar(new int[]{b, c, dilatedH, dilatedW}, value);
                }
            }
        }
    }
    
    return dilated;
  }*/

  public INDArray TranspConv(INDArray input, int stride, int padding){
    return TranspConv(input, stride, padding, true);
  }

  public INDArray TranspConv(INDArray input, int stride, int padding, boolean bias) {
    long[] inputShape = input.shape();
    int batchSize = (int)inputShape[0];
    int inChannels = (int)inputShape[1];
    int inHeight = (int)inputShape[2];
    int inWidth = (int)inputShape[3];
    
    long[] weightShape = W.shape();
    int outChannels = (int)weightShape[0];
    int kernelSize = (int)weightShape[2];
    
    int outHeight = (inHeight - 1) * stride - 2 * padding + kernelSize;
    int outWidth = (inWidth - 1) * stride - 2 * padding + kernelSize;
    
    // Reshape input to [H_in * W_in, C_in]
    INDArray inputReshaped = input.permute(0, 2, 3, 1)
                                  .reshape(batchSize * inHeight * inWidth, inChannels);
    
    // Reshape weights to [C_out * K * K, C_in]  
    INDArray weightReshaped = W.permute(0, 2, 3, 1)
                               .reshape(outChannels * kernelSize * kernelSize, inChannels);
    
    // Matrix multiplication: [H_in * W_in, C_in] × [C_in, C_out * K * K]
    INDArray result = inputReshaped.mmul(weightReshaped.transpose());
    // result: [H_in * W_in, C_out * K * K]
    
    // Reshape to [H_in, W_in, C_out, K, K]
    result = result.reshape(inHeight, inWidth, outChannels, kernelSize, kernelSize);
    
    // Initialize output
    INDArray output = Nd4j.zeros(batchSize, outChannels, outHeight, outWidth);
    
    // Scatter the results to output positions
    for (int h = 0; h < inHeight; h++) {
        for (int w = 0; w < inWidth; w++) {
            int outHStart = h * stride - padding;
            int outWStart = w * stride - padding;
            
            // Boundary checks
            int hStart = Math.max(0, -outHStart);
            int wStart = Math.max(0, -outWStart);
            int hEnd = Math.min(kernelSize, outHeight - outHStart);
            int wEnd = Math.min(kernelSize, outWidth - outWStart);
            
            if (hStart >= hEnd || wStart >= wEnd) continue;
            
            int outHPos = Math.max(outHStart, 0);
            int outWPos = Math.max(outWStart, 0);
            
            for (int cOut = 0; cOut < outChannels; cOut++) {
                INDArray kernelPatch = result.get(NDArrayIndex.point(h), NDArrayIndex.point(w), 
                                                NDArrayIndex.point(cOut),
                                                NDArrayIndex.interval(hStart, hEnd),
                                                NDArrayIndex.interval(wStart, wEnd));
                
                INDArray outPatch = output.get(NDArrayIndex.point(0), NDArrayIndex.point(cOut),
                                             NDArrayIndex.interval(outHPos, outHPos + hEnd - hStart),
                                             NDArrayIndex.interval(outWPos, outWPos + wEnd - wStart));
                
                outPatch.addi(kernelPatch);
            }
        }
    }

    if (bias){
      INDArray summableb = b.reshape(1,outChannels,1,1).broadcast(1,outChannels, outHeight, outHeight);
      output = output.add(summableb);
    }

    return output;
  }

  public INDArray Linear(INDArray input) {
    return Linear(input, true);
  }

  public INDArray Linear(INDArray input, boolean bias) {
    return b.reshape(1, b.shape()[0]).add(input.mmul(W.transpose()));
  }

  public static INDArray concat(INDArray x1, INDArray x2) {
    return Nd4j.concat(1, x1, x2);
  }
  
  public static INDArray SiLU(INDArray x) {
    return x.mul(Transforms.sigmoid(x, true));
  }
  
}
