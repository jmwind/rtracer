use minifb::{Key, Window, WindowOptions};

struct Tupple {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

struct Matrix2 {
    rows: [[f32; 2]; 2],
}

struct Matrix3 {
    rows: [[f32; 3]; 3],
}

struct Matrix4 {
    rows: [[f32; 4]; 4],
}

impl Matrix2 {
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        let val = self.rows.get(row)?.get(col)?;
        Some(*val)
    }
}

impl Matrix3 {
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        let val = self.rows.get(row)?.get(col)?;
        Some(*val)
    }
}

impl Matrix4 {
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        let val = self.rows.get(row)?.get(col)?;
        Some(*val)
    }
}

impl Tupple {
    pub fn vector(x: f64, y: f64, z: f64) -> Tupple {
        Tupple { x, y, z, w: 0.0 }
    }

    pub fn tupple(x: f64, y: f64, z: f64, w: f64) -> Tupple {
        Tupple { x, y, z, w }
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

    pub fn add_r(&self, _other: &Tupple) -> Tupple {
        return Tupple {
            x: self.x + _other.x,
            y: self.y + _other.y,
            z: self.z + _other.z,
            w: self.w + _other.w,
        };
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

    pub fn sub_r(&self, _other: &Tupple) -> Tupple {
        return Tupple {
            x: self.x - _other.x,
            y: self.y - _other.y,
            z: self.z - _other.z,
            w: self.w - _other.w,
        };
    }

    pub fn mul(&mut self, factor: f64) {
        self.x = factor * self.x;
        self.y = factor * self.y;
        self.z = factor * self.z;
        self.w = factor * self.w;
    }

    pub fn mul_r(&self, factor: f64) -> Tupple {
        return Tupple {
            x: factor * self.x,
            y: factor * self.y,
            z: factor * self.z,
            w: factor * self.w,
        };
    }

    pub fn negate(&mut self) {
        self.x = -1.0 * self.x;
        self.y = -1.0 * self.y;
        self.z = -1.0 * self.z;
        self.w = -1.0 * self.w;
    }

    pub fn negate_r(&self) -> Tupple {
        return Tupple {
            x: -1.0 * self.x,
            y: -1.0 * self.y,
            z: -1.0 * self.z,
            w: -1.0 * self.w,
        };
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
        return self.x * _other.x + self.y * _other.y + self.z * _other.z + self.w * _other.w;
    }

    pub fn cross(&self, _other: &Tupple) -> Tupple {
        return Tupple::vector(
            self.y * _other.z - self.z * _other.y,
            self.z * _other.x - self.x * _other.z,
            self.x * _other.y - self.y * _other.x,
        );
    }
}

impl Color {
    // color base 0..1, use new_from_255 for other based initializer
    pub fn new(r: f32, g: f32, b: f32) -> Color {
        return Color { r, g, b };
    }

    pub fn new_from_255(r: u32, g: u32, b: u32) -> Color {
        return Color {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
        };
    }

    pub fn to_u32(&self) -> u32 {
        let (r, g, b) = (
            (self.r as u32 * 255),
            (self.g as u32 * 255),
            (self.b as u32 * 255),
        );
        (r << 16) | (g << 8) | b
    }

    pub fn add(&self, _other: &Color) -> Color {
        return Color::new(self.r + _other.r, self.g + _other.g, self.b + _other.b);
    }

    pub fn mul(&self, _other: &Color) -> Color {
        return Color::new(self.r * _other.r, self.g * _other.g, self.b * _other.b);
    }
}

// ## MAIN ##############################

struct Projectile {
    position: Tupple,
    velocity: Tupple,
}

struct Environment {
    gravity: Tupple,
    wind: Tupple,
}

fn tick(env: Environment, proj: &mut Projectile) {
    // move to new position
    proj.position.add(&proj.velocity);
    // update velocity based on environment
    let mut gravity = env.gravity;
    gravity.add(&env.wind);
    proj.velocity.add(&gravity);
}

const WIDTH: usize = 900;
const HEIGHT: usize = 550;
fn main() {
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = Window::new(
        "RTRACER JML - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));
    window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();

    let mut projectile = Projectile {
        position: Tupple::point(0.0, 1.0, 0.0),
        velocity: Tupple::vector(1.0, 1.4, 0.0).mul_r(6.5),
    };

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for _i in 0..250 {
            let environment = Environment {
                gravity: Tupple::vector(0.0, -0.11, 0.0),
                wind: Tupple::vector(-0.01, 0.0, 0.0),
            };
            tick(environment, &mut projectile);
            let x = projectile.position.x as usize;
            let y = HEIGHT - projectile.position.y as usize;
            if x < WIDTH && y < HEIGHT {
                let index = x + (y * WIDTH);
                buffer[index] = Color::new_from_255(255, 255, 180).to_u32();
            }
            window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
        }
    }
}

// ## TESTS ##############################

#[cfg(test)]
mod tests {
    use crate::*;

    const EPSILON: f64 = 0.00001;
    const POINT: f64 = 1.0;
    const VECTOR: f64 = 0.0;

    fn fequals(a: f64, b: f64) -> bool {
        let n = a - b;
        if n.abs() < EPSILON {
            return true;
        } else {
            return false;
        }
    }

    fn approx_equal(a: f32, b: f32, decimal_places: u8) -> bool {
        let factor = 10.0f32.powi(decimal_places as i32);
        let a = (a * factor).trunc();
        let b = (b * factor).trunc();
        a == b
    }

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

    #[test]
    fn test_mul_color() {
        let c1 = Color::new(1.0, 0.2, 0.4);
        let c2 = Color::new(0.9, 1.0, 0.1);
        let c3 = c1.mul(&c2);
        assert!(approx_equal(c3.r, 0.9, 2));
        assert!(approx_equal(c3.g, 0.2, 2));
        assert!(approx_equal(c3.b, 0.04, 2));
    }

    #[test]
    fn test_matrix_new() {}
}
