package my;

public class CountryFactory {
    
    private CountryFactory() {
    }
    
    private static CountryFactory instance = null;
    
    public static CountryFactory getInstance() {
	if (instance == null) {
	    instance = new CountryFactory();
	}
	return instance;
    }
    
    public ICountry getCountry(String name) {
	if (name.equals(Japan.SYMBOL)) {
	    return new Japan();
	} else if (name.equals(America.SYMBOL)) {
	    return new America();
	} else if (name.equals(China.SYMBOL)) {
	    return new China();
	}
	return null;
    }
}
