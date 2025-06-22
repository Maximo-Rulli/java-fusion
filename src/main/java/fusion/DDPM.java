package fusion;

import java.io.File;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// Core DDPM structure
public class DDPM {
    public UNet model;  // Replace with your actual model class
    private int timesteps;
    private INDArray betas;
    private INDArray alphas;
    private INDArray alphasCumprod;
    
    public DDPM(UNet model, int timesteps) {
        this.model = model;
        this.timesteps = timesteps;
        
        // Create noise schedule using linear beta schedule
        betas = linearBetaSchedule(timesteps);
        
        // alphas = 1.0 - betas
        alphas = Nd4j.ones(timesteps).sub(betas);
        
        // alphas_cumprod = cumulative product of alphas
        alphasCumprod = cumprod(alphas);
    }
    
    // Helper method to create linear beta schedule
    private INDArray linearBetaSchedule(int timesteps) {
        double start = 0.0001;
        double end = 0.02;
        
        // Create linear schedule from start to end
        INDArray schedule = Nd4j.linspace(start, end, timesteps, DataType.FLOAT);
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
    

    public INDArray sample(int saveSteps, int threadNumber) {
        // Start with pure noise
        //INDArray img = Nd4j.randn(1,1,28,28);

        INDArray img = Nd4j.readNpy(new File("./src/main/java/fusion/test_outs/init.npy"));
        
        // Reverse diffusion: go from timestep T-1 down to 0
        for (int i = this.timesteps - 1; i >= 0; i--) {
            System.out.println("Sampling step of thread " + threadNumber + ": " + (timesteps - i) + "/" + timesteps);
            
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
                //System.out.println(img);
                ImageSaver.saveImage(img, "imgs/thr"+ threadNumber +"-steps"+(1500-i));
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