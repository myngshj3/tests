package my;

public class China implements ICountry {
    public static final String SYMBOL = "ch";
    
    public String getName() {
	return "China";
    }
    public String getLocation() {
	return "East Asia";
    }
    
    public int getPopulation() {
	return 80000000;
    }
}
