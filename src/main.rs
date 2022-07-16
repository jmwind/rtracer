
enum TuppleType {
    Point,
    Vector,
}

struct Tupple {
    pub x: u64,
    pub y: u64,
    pub z: u64,
    pub t: TuppleType,
}

impl Tupple {
    pub fn Add(&mut self, other: &Tupple) {

    }

    pub fn Sub(&mut self, other: &Tupple) {

    }
}

fn main() {
    println!("Hello, world!");
    let mut guess = String::from("This is a string");
    let apples = 5;
    let t1 = (1,2,3);
    let t2 = (4,5,6);
    
    print!("this is {} apples of {} and {}", apples, guess, t3);
}
