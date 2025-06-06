package fusion;

// Utility class for image operations and noise handling
public class Utils {
    public static double[] addNoise(double[] image, double[] noise, double alpha) {}

    public static double[] clipImage(double[] image) {}

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