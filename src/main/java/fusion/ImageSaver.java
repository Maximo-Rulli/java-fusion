// I acknowledge that this is not my code, but the visualization is not the main challenge of the project neither

package fusion;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class ImageSaver {
    public static void saveImage(INDArray imageArray, String filename) {
        // Handle different input shapes
        INDArray processed;
        
        if (imageArray.rank() == 4) {
            // [batch, channels, height, width] - take first sample
            processed = imageArray.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()); // [height, width]
        } else {
            throw new IllegalArgumentException("Unsupported array rank: " + imageArray.rank());
        }
        
        // Normalize to 0-255 range
        // Assuming input is in range [-1, 1] or [0, 1]
        INDArray normalized = normalizeForDisplay(processed);
        
        int height = (int) normalized.shape()[0];
        int width = (int) normalized.shape()[1];
        
        // Create grayscale BufferedImage
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        
        // Fill the image
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int grayValue = (int) normalized.getDouble(y, x);
                // Ensure value is in [0, 255] range
                grayValue = Math.max(0, Math.min(255, grayValue));
                
                // For grayscale, RGB values are all the same
                int rgb = (grayValue << 16) | (grayValue << 8) | grayValue;
                image.setRGB(x, y, rgb);
            }
        }
        
        // Save to file
        try {
            ImageIO.write(image, "png", new File(filename + ".png"));
            System.out.println("MNIST image saved to: " + filename + ".png");
        } catch (IOException e) {
            System.err.println("Error saving image: " + e.getMessage());
        }
    }
   
    /**
     * Normalize array values to 0-255 range for display
     */
    private static INDArray normalizeForDisplay(INDArray input) {
        // Find min and max values
        double min = input.minNumber().doubleValue();
        double max = input.maxNumber().doubleValue();
        
        System.out.println("Input range: [" + min + ", " + max + "]");
        
        // Different normalization strategies based on input range
        if (min >= -1.1 && max <= 1.1) {
            // Assume input is in [-1, 1] range (common for DDPM)
            return input.add(1.0).mul(127.5); // Convert [-1,1] to [0,255]
        } else if (min >= -0.1 && max <= 1.1) {
            // Assume input is in [0, 1] range
            return input.mul(255.0); // Convert [0,1] to [0,255]
        } else {
            // General case: normalize to [0,1] then scale to [0,255]
            INDArray normalized = input.sub(min).div(max - min);
            return normalized.mul(255.0);
        }
    }
    
}