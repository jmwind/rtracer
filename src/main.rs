use std::fmt;
use std::fmt::Display;
use minifb::{Key, Window, WindowOptions};

const EPSILON: f64 = 0.00001;
const POINT: f64 = 1.0;
const VECTOR: f64 = 0.0;

struct Tupple {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64
}

impl Tupple {
    pub fn vector(x: f64, y: f64, z: f64) -> Tupple {
        Tupple {
            x,
            y,
            z,
            w: 0.0,
        }
    }

    pub fn tupple(x: f64, y: f64, z: f64, w:f64) -> Tupple {
        Tupple {
            x,
            y,
            z,
            w,
        }
    }

    pub fn point(x: f64, y: f64, z: f64) -> Tupple {
        Tupple {
            x: x,
            y: y,
            z: z,
            w: 1.0,
        }
    }   

    pub fn add(&mut self, _other: &Tupple) {
        self.x = self.x + _other.x;
        self.y = self.y + _other.y;
        self.z = self.z + _other.z;
        let w_val = self.w + _other.w;
        if w_val > 1.0 {            
            panic!("Can't add a point to a point");
        } else {
            self.w = w_val;
        }
    }

    pub fn sub(&mut self, _other: &Tupple) {
        self.x = self.x - _other.x;
        self.y = self.y - _other.y;
        self.z = self.z - _other.z;
        let w_val = self.w - _other.w;
        if w_val >= 0.0 {
            self.w = w_val;
        } else {
            panic!("Can't add substract a point from a vector");
        }
    }

    pub fn mul(&mut self, factor: f64) {
        self.x = factor * self.x;
        self.y = factor * self.y;
        self.z = factor * self.z;
        self.w = factor * self.w;
    }

    pub fn negate(&mut self) {
        self.x = -1.0 * self.x;
        self.y = -1.0 * self.y;
        self.z = -1.0 * self.z;
        self.w = -1.0 * self.w;
    }

    pub fn magnitude(&self) -> f64 {
        let m = self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2);
        return m.abs().sqrt();
    }

    pub fn normalize(&self) -> Tupple {
        let m = self.magnitude();
        return Tupple::tupple(self.x / m, self.y / m, self.z / m, self.w / m);
    }

    pub fn dot(&self, _other: &Tupple) -> f64 {
        return 
            self.x * _other.x +
            self.y * _other.y +
            self.z * _other.z +
            self.w * _other.w;
    }

    pub fn cross(&self, _other: &Tupple) -> Tupple {
        return Tupple::vector(
            self.y * _other.z - self.z * _other.y,            
            self.z * _other.x - self.x * _other.z,
            self.x * _other.y - self.y * _other.x);
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
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

const WIDTH: usize = 900;
const HEIGHT: usize = 550;
fn main() {
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = Window::new(
        "Test - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for i in buffer.iter_mut() {
            *i = 122; // write something more funny here!
        }

        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }
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
        assert_eq!(p1.w, POINT);
        assert!(v1.x == 4.0);
        assert!(v1.y == 2.0);
        assert!(v1.z == 1.0);
        assert_eq!(v1.w, VECTOR);
        assert!(v2.x == -8.0);
        assert!(v2.y == -12.0);
        assert!(v2.z == -11.0);
        assert_eq!(v2.w, VECTOR);
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
        assert_eq!(p1.w, POINT);

        v1.add(&v2);
        assert_eq!(v1.x, -5.0);
        assert_eq!(v1.y, 5.0);
        assert_eq!(v1.z, 9.0);
        assert_eq!(v1.w, VECTOR);
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
        assert_eq!(p1.w, VECTOR);

        p3.sub(&v1);
        assert_eq!(p3.x, -2.0);
        assert_eq!(p3.y, -4.0);
        assert_eq!(p3.z, -6.0);
        assert_eq!(p3.w, POINT);

        v2.sub(&v1);
        assert_eq!(v2.x, -2.0);
        assert_eq!(v2.y, -4.0);
        assert_eq!(v2.z, -6.0);
        assert_eq!(v2.w, VECTOR);
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

    #[test]
    fn test_negate() {        
        let mut v1 = Tupple::tupple(1.0, -2.0, 3.0, -4.0);
        v1.negate();
        assert_eq!(v1.x, -1.0);
        assert_eq!(v1.y, 2.0);
        assert_eq!(v1.z, -3.0);
        assert_eq!(v1.w, 4.0);
    }

    #[test]
    fn test_multiply() {
        let mut v1 = Tupple::tupple(1.0, -2.0, 3.0, -4.0);
        v1.mul(3.5);
        assert_eq!(v1.x, 3.5);
        assert_eq!(v1.y, -7.0);
        assert_eq!(v1.z, 10.5);
        assert_eq!(v1.w, -14.0);

        let mut v2 = Tupple::tupple(1.0, -2.0, 3.0, -4.0);
        v2.mul(0.5);
        assert_eq!(v2.x, 0.5);
        assert_eq!(v2.y, -1.0);
        assert_eq!(v2.z, 1.5);
        assert_eq!(v2.w, -2.0);
    }

    #[test]
    fn test_magnitude() {
        let v1 = Tupple::vector(0.0, 1.0, 0.0);
        let v2 = Tupple::vector(1.0, 0.0, 0.0);
        let v3 = Tupple::vector(0.0, 0.0, 1.0);
        let v4 = Tupple::vector(1.0, 2.0, 3.0);
        let v5 = Tupple::vector(-1.0, -2.0, -3.0);
        assert_eq!(v1.magnitude(), 1.0);
        assert_eq!(v2.magnitude(), 1.0);
        assert_eq!(v3.magnitude(), 1.0);
        assert_eq!(v4.magnitude(), 14.0_f64.sqrt());
        assert_eq!(v5.magnitude(), 14.0_f64.sqrt());

    }

    #[test]
    fn test_normalize() {
        let v1 = Tupple::vector(4.0, 0.0, 0.0).normalize();
        let v2 = Tupple::vector(1.0, 2.0, 3.0).normalize();
        assert_eq!(v1.x, 1.0);
        assert_eq!(v1.y, 0.0);
        assert_eq!(v1.z, 0.0);
        assert_eq!(v2.x, 1.0 / 14.0_f64.sqrt());
        assert_eq!(v2.y, 2.0 / 14.0_f64.sqrt());
        assert_eq!(v2.z, 3.0 / 14.0_f64.sqrt());
    }

    #[test]
    fn test_dot() {
        let v1 = Tupple::vector(1.0, 2.0, 3.0);
        let v2 = Tupple::vector(2.0, 3.0, 4.0);
        let dot = v1.dot(&v2);
        assert_eq!(dot, 20.0);        
    }

    #[test]
    fn test_cross() {
        let v1 = Tupple::vector(1.0, 2.0, 3.0);
        let v2 = Tupple::vector(2.0, 3.0, 4.0);
        let c1 = v1.cross(&v2);
        let c2 = v2.cross(&v1);
        assert_eq!(c1.x, -1.0);
        assert_eq!(c1.y, 2.0);
        assert_eq!(c1.z, -1.0);
        assert_eq!(c2.x, 1.0);
        assert_eq!(c2.y, -2.0);
        assert_eq!(c2.z, 1.0);
    }
}
