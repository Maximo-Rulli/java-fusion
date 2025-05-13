package fusion;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
    public static void main(String[] args) {
        INDArray t = Nd4j.create(new float[]{2});
        System.out.println("Boradcasted " + t.broadcast(1,1,28,28));


        INDArray a = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8}, new int[]{1, 4, 2, 2});
        INDArray b = Nd4j.create(new float[]{-9, -8, -7, -6, -5, -4, -3, -2, -9, -8, -7, -6, -5, -4, -3, -2}, new int[]{1, 4, 2, 2});
        INDArray result = a.add(b);
        System.out.println("Sum:\n" + result);
    }
}
