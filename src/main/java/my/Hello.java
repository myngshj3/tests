package my;

public class Hello {
    public static void main(String[] args) {
	System.out.println("Hello");
	/*
	System.out.println("args.length=" + args.length);
	for (String arg: args) {
	    System.out.println("arg=" + arg);
	}
	if (args.length != 1) {
	    System.out.println("Too few or many argument(s)");
	    return;
	}
	*/
	String symbol = args[0];
	CountryFactory factory = CountryFactory.getInstance();
	ICountry country = factory.getCountry(symbol);
	System.out.println("name: " + country.getName());
	System.out.println("location: " + country.getLocation());
	System.out.println("population: " + country.getPopulation());
    }
}
