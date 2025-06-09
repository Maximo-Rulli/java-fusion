package fusion;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

// Core DDPM structure
public class DDPM {
    public UNet model;  // Replace with your actual model class
    private int timesteps;
    private INDArray betas;
    private INDArray alphas;
    private INDArray alphasCumprod;
    private INDArray sqrtAlphasCumprod;
    private INDArray sqrtOneMinusAlphasCumprod;
    
    public DDPM(UNet model, int timesteps) {
        this.model = model;
        this.timesteps = timesteps;
        
        // Create noise schedule using linear beta schedule
        this.betas = linearBetaSchedule(timesteps);
        
        // alphas = 1.0 - betas
        this.alphas = Nd4j.ones(timesteps).sub(this.betas);
        
        // alphas_cumprod = cumulative product of alphas
        this.alphasCumprod = cumprod(this.alphas);
        
        // Pre-compute values for sampling
        this.sqrtAlphasCumprod = Transforms.sqrt(this.alphasCumprod);
        this.sqrtOneMinusAlphasCumprod = Transforms.sqrt(Nd4j.ones(timesteps).sub(this.alphasCumprod));
    }

    public INDArray qSample(INDArray xStart, INDArray t, INDArray noise) {
        if (noise == null) {
            // Generate random noise with same shape as x_start
            noise = Nd4j.randn(1,1,28,28);
        }
        
        // Get sqrt_alphas_cumprod values for each timestep in the batch
        // t is [batch_size] containing timestep indices, in our case only 1
        INDArray sqrtAlphasCumprodT = gatherByIndices(this.sqrtAlphasCumprod, t);
        INDArray sqrtOneMinusAlphasCumprodT = gatherByIndices(this.sqrtOneMinusAlphasCumprod, t);
        
        // Reshape to broadcast properly
        // This allows element-wise multiplication with [1, channels, height, width]
        sqrtAlphasCumprodT = sqrtAlphasCumprodT.reshape(1, 1, 1, 1);
        sqrtOneMinusAlphasCumprodT = sqrtOneMinusAlphasCumprodT.reshape(1, 1, 1, 1);
        
        // Apply the forward diffusion formula:
        // x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        return sqrtAlphasCumprodT.mul(xStart).add(sqrtOneMinusAlphasCumprodT.mul(noise));
    }
    
    // Helper method to create linear beta schedule
    private INDArray linearBetaSchedule(int timesteps) {
        double start = 0.0001;
        double end = 0.02;
        
        // Create linear schedule from start to end
        INDArray schedule = Nd4j.linspace(DataType.FLOAT, start, end, (long) timesteps);
        return schedule;
    }
    
    // Helper method to compute cumulative product
    private INDArray cumprod(INDArray input) {
        INDArray result = Nd4j.zeros(input.shape());
        result.putScalar(0, input.getDouble(0));
        
        for (int i = 1; i < input.length(); i++) {
            result.putScalar(i, result.getDouble(i-1) * input.getDouble(i));
        }
        
        return result;
    }
    
    // Helper method to gather values by indices (similar to PyTorch's indexing)
    private INDArray gatherByIndices(INDArray source, INDArray indices) {
        INDArray result = Nd4j.zeros(indices.shape());
        
        for (int i = 0; i < indices.length(); i++) {
            int index = indices.getInt(i);
            result.putScalar(i, source.getDouble(index));
        }
        
        return result;
    }

    public INDArray sample(int saveSteps) {
        // Start with pure noise
        INDArray img = Nd4j.randn(1,1,28,28);
        
        // Reverse diffusion: go from timestep T-1 down to 0
        for (int i = this.timesteps - 1; i >= 0; i--) {
            System.out.println("Sampling step: " + (this.timesteps - i) + "/" + this.timesteps);
            
            // Create timestep tensor [batch_size] filled with current timestep
            INDArray t = Nd4j.zeros(1).addi(i);
            
            // Predict noise using the model
            INDArray predictedNoise = model.predict(img, t);
            
            // Get coefficients for current timestep
            double alpha = this.alphas.getDouble(i);
            double alphaCumprod = this.alphasCumprod.getDouble(i);
            double beta = this.betas.getDouble(i);
            
            // Compute the denoising step using DDPM reverse process formula:
            // x_{t-1} = (1/sqrt(alpha_t)) * (x_t - ((1-alpha_t)/sqrt(1-alpha_cumprod_t)) * predicted_noise)
            
            // First term: 1/sqrt(alpha_t)
            double oneOverSqrtAlpha = 1.0 / Math.sqrt(alpha);
            
            // Second term: (1-alpha_t)/sqrt(1-alpha_cumprod_t)
            double noiseCoeff = (1.0 - alpha) / Math.sqrt(1.0 - alphaCumprod);
            
            // Apply the denoising formula
            INDArray denoised = img.sub(predictedNoise.mul(noiseCoeff));
            img = denoised.mul(oneOverSqrtAlpha);

            if (i%saveSteps == 0){
                ImageSaver.saveImage(img, "output-"+i+".png");
            }
            
            // Add noise for all steps except the last one (i > 0)
            if (i > 0) {
                INDArray noise = Nd4j.randn(img.shape());
                img = img.add(noise.mul(Math.sqrt(beta)));
            }
        }
        
        return img;
    }
}