package my;

public class America implements ICountry {
    public static final String SYMBOL = "us";
    
    public String getName() {
	return "America";
    }
    public String getLocation() {
	return "North America";
    }
    
    public int getPopulation() {
	return 84000000;
    }
}
