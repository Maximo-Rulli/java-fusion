package fusion;

import java.io.File;
import java.util.HashMap;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// Core DDPM structure
public class DDPM {
    private BetaSchedule betaSchedule;
    private double[] betas;
    private double[] alphas;
    private double[] alphaCumprod;
    private int timesteps;
    private UNet model;

    public DDPM(int timesteps) {
        System.out.println();
    }

    public void sampleNoise(int[] shape) {}

    public void qSample(double[] x0, double[] noise, int t) {}

    public void pSample(double[] xt, int t) {}

    public void generateSamples(int numSamples) {}
}