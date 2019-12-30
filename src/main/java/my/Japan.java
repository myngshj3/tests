package my;

//import ICountry;

public class Japan implements ICountry {
    public static final String SYMBOL = "jp";
    
    public String getName() {
	return "Japan";
    }
    public String getLocation() {
	return "East Asia";
    }
    
    public int getPopulation() {
	return 12000000;
    }
}
