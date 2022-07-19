use std::fmt;
use std::fmt::Display;

const EPSILON: f64 = 0.00001;

enum TuppleType {
    Vector = 0,
    Point,
}

struct Tupple {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub t: TuppleType,
}

impl Tupple {
    pub fn vector(x: f64, y: f64, z: f64) -> Tupple {
        Tupple {
            x: x,
            y: y,
            z: z,
            t: TuppleType::Vector,
        }
    }

    pub fn point(x: f64, y: f64, z: f64) -> Tupple {
        Tupple {
            x: x,
            y: y,
            z: z,
            t: TuppleType::Point,
        }
    }

    fn type_value(&self) -> i32 {
        match self.t {
            TuppleType::Vector => 0,
            TuppleType::Point => 1,
        }
    }

    fn set_type_from_value(&mut self, t_val: i32) {
        match t_val {
            0 => self.t = TuppleType::Vector,
            1 => self.t = TuppleType::Point,
            _ => panic!("Invalid type value"),
        }
    }

    pub fn add(&mut self, _other: &Tupple) {
        self.x = self.x + _other.x;
        self.y = self.y + _other.y;
        self.z = self.z + _other.z;
        let t_val = self.type_value() + _other.type_value();
        if t_val < 2 {
            self.set_type_from_value(t_val);
        } else {
            panic!("Can't add a point to a point");
        }
    }

    pub fn sub(&mut self, _other: &Tupple) {
        self.x = self.x - _other.x;
        self.y = self.y - _other.y;
        self.z = self.z - _other.z;
        let t_val = self.type_value() - _other.type_value();
        if t_val >= 0 {
            self.set_type_from_value(t_val);
        } else {
            panic!("Can't add substract a point from a vector");
        }
    }

    pub fn negate(&mut self) {
        self.x = -1.0 * self.x;
        self.y = -1.0 * self.y;
        self.z = -1.0 * self.z;
    }
}

fn fequals(a: f64, b: f64) -> bool {
    let n = a - b;
    if n.abs() < EPSILON {
        return true;
    } else {
        return false;
    }
}

impl Display for Tupple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_string = match self.t {
            TuppleType::Point => "Point",
            TuppleType::Vector => "Vector",
        };
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, type_string)
    }
}

fn main() {
    let p1 = Tupple::point(0.0, 1.0, 1.0);
    let p2: Tupple = Tupple::point(0.0, 1.0, 1.0);
    let v1 = Tupple::vector(4.0, 2.0, 1.0);
    let v2 = Tupple::vector(8.0, 12.0, 11.0);

    println!("p1 {} p2 {} p3 {} p4 {}", p1, p2, v1, v2);
}

// ## TESTS ##############################

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn tupple_create() {
        let p1 = Tupple::point(0.0, 1.0, 1.0);
        let v1 = Tupple::vector(4.0, 2.0, 1.0);
        let v2 = Tupple::vector(-8.0, -12.0, -11.0);
        assert!(p1.x == 0.0);
        assert!(p1.y == 1.0);
        assert!(p1.z == 1.0);
        assert!(matches!(p1.t, TuppleType::Point));
        assert!(v1.x == 4.0);
        assert!(v1.y == 2.0);
        assert!(v1.z == 1.0);
        assert!(matches!(v1.t, TuppleType::Vector));
        assert!(v2.x == -8.0);
        assert!(v2.y == -12.0);
        assert!(v2.z == -11.0);
        assert!(matches!(v2.t, TuppleType::Vector));
    }

    #[test]
    fn test_floating_equals() {
        let n1 = 0.1;
        let n2 = 2.0;
        let n3 = 2.0000001;
        let n4 = 0.05;
        let n5 = 0.050000001;
        assert!(fequals(n2, n3));
        assert!(!fequals(n1, n2));
        assert!(!fequals(n1, n4));
        assert!(fequals(n4, n5));
    }

    #[test]
    fn test_tupple_additions() {
        let mut p1 = Tupple::point(3.0, -2.0, 5.0);
        let mut v1 = Tupple::vector(-2.0, 3.0, 1.0);
        let v2 = Tupple::vector(-3.0, 2.0, 8.0);

        p1.add(&v1);
        assert_eq!(p1.x, 1.0);
        assert_eq!(p1.y, 1.0);
        assert_eq!(p1.z, 6.0);
        assert!(matches!(p1.t, TuppleType::Point));

        v1.add(&v2);
        assert_eq!(v1.x, -5.0);
        assert_eq!(v1.y, 5.0);
        assert_eq!(v1.z, 9.0);
        assert!(matches!(v1.t, TuppleType::Vector));
    }

    #[test]
    fn test_tupple_substractions() {
        let mut p1 = Tupple::point(3.0, 2.0, 1.0);
        let p2 = Tupple::point(5.0, 6.0, 7.0);
        let v1 = Tupple::vector(5.0, 6.0, 7.0);
        let mut p3 = Tupple::point(3.0, 2.0, 1.0);
        let mut v2 = Tupple::vector(3.0, 2.0, 1.0);

        p1.sub(&p2);
        assert_eq!(p1.x, -2.0);
        assert_eq!(p1.y, -4.0);
        assert_eq!(p1.z, -6.0);
        assert!(matches!(p1.t, TuppleType::Vector));

        p3.sub(&v1);
        assert_eq!(p3.x, -2.0);
        assert_eq!(p3.y, -4.0);
        assert_eq!(p3.z, -6.0);
        assert!(matches!(p3.t, TuppleType::Point));

        v2.sub(&v1);
        assert_eq!(v2.x, -2.0);
        assert_eq!(v2.y, -4.0);
        assert_eq!(v2.z, -6.0);
        assert!(matches!(v2.t, TuppleType::Vector));
    }

    #[test]
    #[should_panic]
    fn test_tupple_additions_exceptions() {
        let mut p1 = Tupple::point(3.0, -2.0, 5.0);
        let p2 = Tupple::point(-2.0, 3.0, 1.0);
        p1.add(&p2);
    }

    #[test]
    #[should_panic]
    fn test_tupple_subs_exceptions() {
        let p1 = Tupple::point(3.0, -2.0, 5.0);
        let mut v1 = Tupple::vector(-2.0, 3.0, 1.0);
        v1.sub(&p1);
    }
}
