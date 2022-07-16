use std::fmt::Display;
use std::fmt;

enum TuppleType {
    Point = 1,
    Vector,
}

struct Tupple {
    pub x: u64,
    pub y: u64,
    pub z: u64,
    pub t: TuppleType,
}

impl Tupple {
    pub fn vector(x:u64, y:u64, z:u64) -> Tupple {
        Tupple {x: x, y: y, z: z, t: TuppleType::Vector}
    }

    pub fn point(x:u64, y:u64, z:u64) -> Tupple {
        Tupple {x: x, y: y, z: z, t: TuppleType::Point}
    }

    pub fn add(&mut self, other: &Tupple) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
        self.z = self.z + other.z;

    }

    pub fn sub(&mut self, other: &Tupple) {

    }
}

impl Display for Tupple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_string = match self.t { TuppleType::Point => "Point", TuppleType::Vector => "Vector" };
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, type_string)
    } 
}

fn main() {    
    let p1 = Tupple::point(0, 1, 1);
    let p2: Tupple = Tupple::point(0, 1, 1);
    let v1 = Tupple::vector(4, 2, 1);
    let v2 = Tupple::vector(8, 12, 11);

    println!("p1 {} p2 {} p3 {} p4 {}", p1,p2,v1,v2);
}
