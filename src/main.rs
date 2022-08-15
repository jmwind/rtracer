use minifb::{Key, Window, WindowOptions};
use std::ops;
use std::sync::atomic::{AtomicUsize, Ordering};

const PI: f64 = 3.141592653589793238462643383279502884197169399375105;

macro_rules! point {
    ($x:expr, $y:expr, $z:expr) => {
        Tupple::point($x as f64, $y as f64, $z as f64)
    };
}

macro_rules! vector {
    ($x:expr, $y:expr, $z:expr) => {
        Tupple::vector($x as f64, $y as f64, $z as f64)
    };
}
macro_rules! ray {
    ($point:expr, $vector:expr) => {
        Ray::new($point, $vector);
    };
}

macro_rules! rotation_x {
    ($radians:expr) => {
        Matrix::<f64>::x_rotation($radians);
    };
}

macro_rules! rotation_y {
    ($radians:expr) => {
        Matrix::<f64>::y_rotation($radians);
    };
}

macro_rules! rotation_z {
    ($radians:expr) => {
        Matrix::<f64>::z_rotation($radians);
    };
}

macro_rules! scaling {
    ($x:expr, $y:expr, $z:expr) => {
        Matrix::<f64>::scaling($x as f64, $y as f64, $z as f64);
    };
}

macro_rules! translation {
    ($x:expr, $y:expr, $z:expr) => {
        Matrix::<f64>::translation($x as f64, $y as f64, $z as f64);
    };
}

fn radians(degrees: f64) -> f64 {
    return (degrees / 180.0) * PI;
}

struct Canvas {
    pub buffer: Vec<u32>,
    pub window: Window,
    pub width: usize,
    pub height: usize,
}

#[derive(PartialEq, Debug, Clone)]
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

struct Ray {
    pub origin: Tupple,
    pub direction: Tupple,
}

struct Sphere {
    pub id: usize,
    pub center: Tupple,
    pub radius: f64,
}

static COUNTER: AtomicUsize = AtomicUsize::new(1);
fn get_id() -> usize {
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl Sphere {
    pub fn new() -> Sphere {
        Sphere {
            id: get_id(),
            center: point!(0, 0, 0),
            radius: 1.0,
        }
    }

    pub fn intersect(&self, ray: Ray) -> Vec<f64> {
        let mut intersections = vec![];

        let sphere_to_ray = ray.origin.sub_r(&self.center);
        let a = ray.direction.dot(&ray.direction);
        let b = 2.0 * ray.direction.dot(&sphere_to_ray);
        let c = sphere_to_ray.dot(&sphere_to_ray) - 1.0;

        let discriminant = b.powi(2) - 4.0 * a * c;

        if discriminant >= 0.0 {
            intersections.push((-b - discriminant.sqrt()) / 2.0 * a);
            intersections.push((-b + discriminant.sqrt()) / 2.0 * a);
        }

        return intersections;
    }
}

impl Ray {
    pub fn new(origin: Tupple, direction: Tupple) -> Ray {
        Ray { origin, direction }
    }

    pub fn position(&self, distance: f64) -> Tupple {
        return self.direction.mul_r(distance).add_r(&self.origin);
    }
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Canvas {
        let buffer: Vec<u32> = vec![0; width * height];
        let mut window = Window::new(
            "RTRACER JML - ESC to exit",
            width,
            height,
            WindowOptions::default(),
        )
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });

        // Limit to max ~60 fps update rate
        window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
        Canvas {
            buffer,
            window,
            width,
            height,
        }
    }

    pub fn index(&self, x: usize, y: usize) -> usize {
        return x + (y * self.width);
    }

    pub fn point(&mut self, point: Tupple, color: Color, diameter: usize) {
        let color_u32 = color.to_u32();
        let index = self.index(point.x as usize, point.y as usize);
        self.buffer[index] = color_u32;
        for d in 0..diameter {
            if diameter > 1 {
                self.buffer[index + d] = color_u32;
                self.buffer[index - d] = color_u32;
                self.buffer[index + (self.width + d)] = color_u32;
                self.buffer[index + (self.width - d)] = color_u32;
            }
        }
    }

    pub fn open(&mut self) {
        fn empty(c: &Canvas) {}
        self.open_redraw(&empty);
    }

    pub fn open_redraw(&mut self, f: &dyn Fn(&Canvas)) {
        while self.window.is_open() && !self.window.is_key_down(Key::Escape) {
            f(&self);
            self.window
                .update_with_buffer(&self.buffer, self.width, self.height)
                .unwrap();
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Matrix<
    T: Copy
        + PartialEq
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>,
> {
    data: Vec<T>,
    n_rows: usize,
    n_cols: usize,
}

impl<
        T: Copy
            + PartialEq
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Div<Output = T>,
    > Matrix<T>
{
    pub fn new(mut v: Vec<Vec<T>>) -> Matrix<T> {
        let n_rows = v.len();
        let n_cols = v[0].len();
        let mut data = vec![];
        for row in v.iter_mut() {
            while row.len() > 0 {
                data.push(row.remove(0));
            }
        }
        Matrix {
            data,
            n_rows,
            n_cols,
        }
    }

    /// Creates a matrix by using the 1-d vector. The matrix is 'filled' row after row
    /// from the linear data structure.
    pub fn create_from_data(data: Vec<T>, n_rows: usize, n_cols: usize) -> Matrix<T> {
        if data.len() != n_rows * n_cols {
            panic!("not compatible dimension!");
        }
        Matrix {
            data,
            n_rows,
            n_cols,
        }
    }

    pub fn create_column_from_data(data: Vec<T>) -> Matrix<T> {
        let n_rows = data.len();
        Matrix {
            data: data,
            n_rows: n_rows,
            n_cols: 1,
        }
    }

    pub fn identity_f32() -> Matrix<f32> {
        return Matrix::new(vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);
    }

    pub fn identity_i32() -> Matrix<i32> {
        return Matrix::new(vec![
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
        ]);
    }

    pub fn translation(x: f64, y: f64, z: f64) -> Matrix<f64> {
        return Matrix::new(vec![
            vec![1.0, 0.0, 0.0, x],
            vec![0.0, 1.0, 0.0, y],
            vec![0.0, 0.0, 1.0, z],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);
    }

    pub fn scaling(x: f64, y: f64, z: f64) -> Matrix<f64> {
        return Matrix::new(vec![
            vec![x, 0.0, 0.0, 0.0],
            vec![0.0, y, 0.0, 0.0],
            vec![0.0, 0.0, z, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);
    }

    pub fn x_rotation(r: f64) -> Matrix<f64> {
        return Matrix::new(vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, r.cos(), -r.sin(), 0.0],
            vec![0.0, r.sin(), r.cos(), 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);
    }

    pub fn y_rotation(r: f64) -> Matrix<f64> {
        return Matrix::new(vec![
            vec![r.cos(), 0.0, r.sin(), 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![-r.sin(), 0.0, r.cos(), 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);
    }

    pub fn z_rotation(r: f64) -> Matrix<f64> {
        return Matrix::new(vec![
            vec![r.cos(), -r.sin(), 0.0, 0.0],
            vec![r.sin(), r.cos(), 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);
    }

    pub fn shearing(xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Matrix<f64> {
        return Matrix::new(vec![
            vec![1.0, xy, xz, 0.0],
            vec![yx, 1.0, yz, 0.0],
            vec![zx, zy, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);
    }

    pub fn determinant_i32(m: &Matrix<i32>) -> i32 {
        return m.determinant(0, -1);
    }

    pub fn determinant_f32(m: &Matrix<f32>) -> f32 {
        return m.determinant(0.0, -1.0);
    }

    /// Obtain the element at row `i` and column `j`.
    pub fn get(&self, i: usize, j: usize) -> T {
        return self.data[i * self.n_cols + j];
    }

    /// Obtain the `i`'th row as a 1-d matrix.
    pub fn get_row(&self, i: usize) -> Matrix<T> {
        let mut row = vec![];
        for e in &self.data[i * self.n_cols..(i + 1) * self.n_cols] {
            row.push(*e);
        }
        let n_cols = row.len();
        Matrix::create_from_data(row, 1, n_cols)
    }

    /// Obtain the `i`'th column as a 1-d matrix.
    pub fn get_col(&self, i: usize) -> Matrix<T> {
        let col = self
            .iter()
            .filter(|(_, _, col)| *col == i)
            .map(|(e, _, _)| *e)
            .collect::<Vec<T>>();
        let n_rows = col.len();
        Matrix::create_from_data(col, n_rows, 1)
    }

    pub fn submatrix(&self, row: usize, col: usize) -> Matrix<T> {
        let mut data = vec![];
        for row_num in 0..self.n_rows {
            for col_num in 0..self.n_cols {
                if row_num != row && col_num != col {
                    data.push(self.data[col_num + row_num * self.n_cols]);
                }
            }
        }
        Matrix::create_from_data(data, self.n_cols - 1, self.n_rows - 1)
    }

    /// Create an immutable iterator over the matrix.
    /// The iteration is performed row after row.
    pub fn iter<'a>(&'a self) -> MatrixIterator<'a, T> {
        MatrixIterator::new(0, 0, 0, &self.data, self.n_rows, self.n_cols)
    }

    /// Create an mutable iterator over the matrix.
    /// The iteration is performed row after row.
    pub fn iter_mut<'a>(&'a mut self) -> MatrixIteratorMut<'a, T> {
        MatrixIteratorMut::new(0, 0, &mut self.data, self.n_rows, self.n_cols)
    }

    /// Transposes a copy of the matrix and returns the result.
    pub fn transpose(&self) -> Matrix<T> {
        let mut data = vec![];
        for j in 0..self.n_cols {
            for i in 0..self.n_rows {
                data.push(self.data[j + i * self.n_cols]);
            }
        }
        Matrix::create_from_data(data, self.n_cols, self.n_rows)
    }

    pub fn determinant(&self, initial: T, transform: T) -> T {
        let mut determinant = initial;
        if self.data.len() == 4 {
            let ad = self.get(0, 0) * self.get(1, 1);
            let bc = self.get(0, 1) * self.get(1, 0);
            determinant = ad - bc;
        } else {
            for col in 0..self.n_cols {
                determinant =
                    determinant + self.get(0, col) * self.cofactor(0, col, initial, transform);
            }
        }
        return determinant;
    }

    pub fn inverse(&self, initial: T, transform: T) -> Matrix<T> {
        let d = self.determinant(initial, transform);
        if d == transform {
            panic!("not invertable");
        }

        let mut data = vec![];
        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                let c = self.cofactor(row, col, initial, transform);
                data.push(c / d);
            }
        }
        return Matrix::create_from_data(data, self.n_cols, self.n_rows).transpose();
    }

    pub fn minor(&self, row: usize, col: usize, initial: T, transform: T) -> T {
        let m = self.submatrix(row, col);
        return m.determinant(initial, transform);
    }

    pub fn cofactor(&self, row: usize, col: usize, initial: T, transform: T) -> T {
        let minor = self.minor(row, col, initial, transform);
        if (row + col) % 2 == 0 {
            return minor;
        } else {
            return minor * transform;
        }
    }
}

impl<
        T: Copy
            + PartialEq
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Div<Output = T>,
    > ops::Mul<&Matrix<T>> for &Matrix<T>
{
    type Output = Matrix<T>;
    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        // result will always be the smaller size vector
        let mut it_cols = self.n_cols;
        if it_cols > other.n_cols {
            it_cols = other.n_cols;
        }

        let mut data = vec![];
        for r in 0..self.n_rows {
            for c in 0..it_cols {
                let mut other_column = c;
                if other.n_cols != self.n_cols {
                    other_column = 0;
                }
                let new_value = self.get(r, 0) * other.get(0, other_column)
                    + self.get(r, 1) * other.get(1, other_column)
                    + self.get(r, 2) * other.get(2, other_column)
                    + self.get(r, 3) * other.get(3, other_column);
                data.push(new_value);
            }
        }
        return Matrix::create_from_data(data, self.n_rows, it_cols);
    }
}

pub struct MatrixIteratorMut<'a, T: Copy> {
    row_idx: usize,
    col_idx: usize,
    data: &'a mut [T],
    n_rows: usize,
    n_cols: usize,
}

impl<'a, T: Copy> MatrixIteratorMut<'a, T> {
    pub fn new(
        row_idx: usize,
        col_idx: usize,
        data: &'a mut [T],
        n_rows: usize,
        n_cols: usize,
    ) -> MatrixIteratorMut<'a, T> {
        MatrixIteratorMut {
            row_idx,
            col_idx,
            data,
            n_rows,
            n_cols,
        }
    }
}

impl<'a, T: Copy> std::iter::Iterator for MatrixIteratorMut<'a, T> {
    type Item = (&'a mut T, usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.col_idx == self.n_cols {
            self.row_idx += 1;
            self.col_idx = 0;
        }
        return if self.row_idx == self.n_rows {
            None
        } else {
            let col_idx = self.col_idx;
            self.col_idx += 1;
            let data = std::mem::replace(&mut self.data, &mut []);
            if let Some((v, rest)) = data.split_first_mut() {
                self.data = rest;
                Some((v, self.row_idx, col_idx))
            } else {
                None
            }
        };
    }
}

pub struct MatrixIterator<'a, T: Copy> {
    row_idx: usize,
    col_idx: usize,
    idx: usize,
    data: &'a [T],
    n_rows: usize,
    n_cols: usize,
}

impl<'a, T: Copy> MatrixIterator<'a, T> {
    pub fn new(
        row_idx: usize,
        col_idx: usize,
        idx: usize,
        data: &'a [T],
        n_rows: usize,
        n_cols: usize,
    ) -> MatrixIterator<'a, T> {
        MatrixIterator {
            row_idx,
            col_idx,
            idx,
            data,
            n_rows,
            n_cols,
        }
    }
}

impl<'a, T: Copy> std::iter::Iterator for MatrixIterator<'a, T> {
    type Item = (&'a T, usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.col_idx == self.n_cols {
            self.row_idx += 1;
            self.col_idx = 0;
        }
        return if self.row_idx == self.n_rows {
            None
        } else {
            let col_idx = self.col_idx;
            let idx = self.idx;
            self.col_idx += 1;
            self.idx += 1;
            Some((&self.data[idx], self.row_idx, col_idx))
        };
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

    pub fn translate(&self, matrix: &Matrix<f64>) -> Tupple {
        let m = Matrix::new(vec![vec![self.x], vec![self.y], vec![self.z], vec![self.w]]);
        let translation = matrix * &m;
        return Tupple::tupple(
            translation.get(0, 0),
            translation.get(1, 0),
            translation.get(2, 0),
            translation.get(3, 0),
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

const WIDTH: usize = 900;
const HEIGHT: usize = 550;
fn main() {
    let mut canvas = Canvas::new(WIDTH, HEIGHT);

    let offset = point!(WIDTH / 2, HEIGHT / 2, 0);
    let radius = 200.0;
    let six = point!(0, 1, 0);

    for hour in 0..12 {
        let rotate = rotation_z!(radians(hour as f64 * 30.0));
        let point = six.translate(&rotate).mul_r(radius).add_r(&offset);
        canvas.point(point, Color::new_from_255(255, 255, 0), 3);
    }

    canvas.open();
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

    fn fmatrix_equals(m1: Matrix<f64>, m2: Matrix<f64>) -> bool {
        for elem in m1.iter() {
            if !fequals(*elem.0, m2.get(elem.1, elem.2)) {
                return false;
            }
        }
        return true;
    }

    fn ftupple_equals(t1: Tupple, t2: Tupple) -> bool {
        return fequals(t1.x, t2.x)
            && fequals(t1.y, t2.y)
            && fequals(t1.z, t2.z)
            && fequals(t1.w, t2.w);
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
    fn test_matrix_iter() {
        let m = Matrix::new(vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(
            m.iter().collect::<Vec<(&u32, usize, usize)>>(),
            vec![(&0, 0, 0), (&1, 0, 1), (&2, 1, 0), (&3, 1, 1)]
        );
    }

    #[test]
    fn test_matrix_iter_mut() {
        let mut m = Matrix::new(vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(
            m.iter_mut().collect::<Vec<(&mut u32, usize, usize)>>(),
            vec![
                (&mut 0, 0, 0),
                (&mut 1, 0, 1),
                (&mut 2, 1, 0),
                (&mut 3, 1, 1)
            ]
        );
    }

    #[test]
    fn test_matrix_get_row_1() {
        let m = Matrix::new(vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(m.get_row(0), Matrix::create_from_data(vec![0, 1], 1, 2));
    }

    #[test]
    fn test_matrix_get_row_1f() {
        let m = Matrix::new(vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0],
        ]);
        assert_eq!(
            m.get_row(0),
            Matrix::create_from_data(vec![0.0, 1.0, 2.0], 1, 3)
        );
        assert_eq!(
            m.get_row(1),
            Matrix::create_from_data(vec![3.0, 4.0, 5.0], 1, 3)
        );
        assert_eq!(
            m.get_row(2),
            Matrix::create_from_data(vec![6.0, 7.0, 8.0], 1, 3)
        );
    }

    #[test]
    fn test_matrix_get_row_2() {
        let m = Matrix::new(vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(m.get_row(1), Matrix::create_from_data(vec![2, 3], 1, 2));
    }

    #[test]
    fn test_matrix_get_col_1() {
        let m = Matrix::new(vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(m.get_col(0), Matrix::create_from_data(vec![0, 2], 2, 1));
    }

    #[test]
    fn test_matrix_get_col_2() {
        let m = Matrix::new(vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(m.get_col(1), Matrix::create_from_data(vec![1, 3], 2, 1));
    }

    #[test]
    fn test_matrix_trans() {
        let m = Matrix::new(vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(m.transpose(), Matrix::new(vec![vec![0, 2], vec![1, 3]]));

        let identity: Matrix<i32> = Matrix::<i32>::identity_i32();
        assert_eq!(identity.transpose(), Matrix::<i32>::identity_i32());
    }

    #[test]
    fn test_matrix_equality() {
        let m1 = Matrix::new(vec![vec![0, 1], vec![2, 3]]);
        let m2 = Matrix::new(vec![vec![0, 1], vec![2, 3]]);
        let m3 = Matrix::new(vec![vec![0, 1], vec![6, 3]]);
        assert!(m1 == m2);
        assert_ne!(m1, m3);
    }

    #[test]
    fn test_matrix_multiply() {
        let m1 = Matrix::new(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 8, 7, 6],
            vec![5, 4, 3, 2],
        ]);
        let m2 = Matrix::new(vec![
            vec![-2, 1, 2, 3],
            vec![3, 2, 1, -1],
            vec![4, 3, 6, 5],
            vec![1, 2, 7, 8],
        ]);
        let m3 = Matrix::new(vec![
            vec![20, 22, 50, 48],
            vec![44, 54, 114, 108],
            vec![40, 58, 110, 102],
            vec![16, 26, 46, 42],
        ]);
        let result = &m1 * &m2;
        assert!(result == m3);

        let m4 = Matrix::new(vec![
            vec![1, 2, 3, 4],
            vec![2, 4, 4, 2],
            vec![8, 6, 4, 1],
            vec![0, 0, 0, 1],
        ]);

        let m5 = Matrix::create_column_from_data(vec![1, 2, 3, 1]);
        let result2 = Matrix::create_column_from_data(vec![18, 24, 33, 1]);

        let m6 = &m4 * &m5;
        assert_eq!(m6, result2);
    }

    #[test]
    fn test_matrix_identity() {
        let m1 = Matrix::new(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 8, 7, 6],
            vec![5, 4, 3, 2],
        ]);
        let m2 = Matrix::new(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 8, 7, 6],
            vec![5, 4, 3, 2],
        ]);
        let identity: Matrix<i32> = Matrix::<i32>::identity_i32();

        let result = &m1 * &identity;
        assert_eq!(m2, result);

        let m11 = Matrix::new(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 8.0, 7.0, 6.0],
            vec![5.0, 4.0, 3.0, 2.0],
        ]);
        let m21 = Matrix::new(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 8.0, 7.0, 6.0],
            vec![5.0, 4.0, 3.0, 2.0],
        ]);

        let identity: Matrix<f32> = Matrix::<f32>::identity_f32();

        let result2 = &m11 * &identity;
        assert_eq!(m21, result2);
    }

    #[test]
    fn test_matrix_determinant() {
        let m1 = Matrix::new(vec![vec![1, 5], vec![-3, 2]]);
        assert_eq!(Matrix::<i32>::determinant_i32(&m1), 17);

        let m1 = Matrix::new(vec![vec![1.0, 5.0], vec![-3.0, 2.0]]);
        assert_eq!(Matrix::<f32>::determinant_f32(&m1), 17.0);
    }

    #[test]
    fn test_matrix_submatrix() {
        let m1 = Matrix::new(vec![vec![1, 5, 0], vec![-3, 2, 7], vec![0, 6, 3]]);
        let m1_result = Matrix::new(vec![vec![-3, 2], vec![0, 6]]);
        let m2 = Matrix::new(vec![
            vec![-6, 1, 1, 6],
            vec![-8, 5, 8, 6],
            vec![-1, 0, 8, 2],
            vec![-7, 1, -1, 1],
        ]);
        let m2_result = Matrix::new(vec![vec![-6, 1, 6], vec![-8, 8, 6], vec![-7, -1, 1]]);
        assert_eq!(m1.submatrix(0, 2), m1_result);
        assert_eq!(m2.submatrix(2, 1), m2_result);
    }

    #[test]
    fn test_matrix_minor() {
        let m = Matrix::new(vec![vec![3, 5, 0], vec![2, -1, -7], vec![6, -1, 5]]);
        assert_eq!(m.minor(1, 0, 0, -1), 25);
    }

    #[test]
    fn test_matrix_cofactor() {
        let m = Matrix::new(vec![vec![3, 5, 0], vec![2, -1, -7], vec![6, -1, 5]]);
        assert_eq!(m.cofactor(1, 0, 0, -1), -25);
    }

    #[test]
    fn test_matrix_determinant_large() {
        let m = Matrix::new(vec![vec![1, 2, 6], vec![-5, 8, -4], vec![2, 6, 4]]);
        assert!(m.cofactor(0, 0, 0, -1) == 56);
        assert!(m.cofactor(0, 1, 0, -1) == 12);
        assert!(m.cofactor(0, 2, 0, -1) == -46);
        assert!(m.determinant(0, -1) == -196);

        let m = Matrix::new(vec![
            vec![-2, -8, 3, 5],
            vec![-3, 1, 7, 3],
            vec![1, 2, -9, 6],
            vec![-6, 7, 7, -9],
        ]);
        assert!(m.determinant(0, -1) == -4071);
    }

    #[test]
    fn test_matrix_inversion() {
        let m = Matrix::new(vec![
            vec![6, 4, 4, 4],
            vec![5, 5, 7, 6],
            vec![4, -9, 3, -7],
            vec![9, 1, 7, -6],
        ]);
        assert!(Matrix::<i32>::determinant_i32(&m) == -2120);
        let m = Matrix::new(vec![
            vec![-4, 2, -2, -3],
            vec![9, 6, 2, 6],
            vec![0, -5, 1, -5],
            vec![0, 0, 0, 0],
        ]);
        assert!(Matrix::<i32>::determinant_i32(&m) == 0);
        let m = Matrix::new(vec![
            vec![-5.0, 2.0, 6.0, -8.0],
            vec![1.0, -5.0, 1.0, 8.0],
            vec![7.0, 7.0, -6.0, -7.0],
            vec![1.0, -3.0, 7.0, 4.0],
        ]);
        let m_result = Matrix::new(vec![
            vec![0.21805, 0.45113, 0.24060, -0.04511],
            vec![-0.80827, -1.45677, -0.44361, 0.52068],
            vec![-0.07895, -0.22368, -0.05263, 0.19737],
            vec![-0.52256, -0.81391, -0.30075, 0.30639],
        ]);
        let new_m = m.inverse(0.0, -1.0);
        assert!(fmatrix_equals(m_result, new_m));

        let m = Matrix::new(vec![
            vec![8.0, -5.0, 9.0, 2.0],
            vec![7.0, 5.0, 6.0, 1.0],
            vec![-6.0, 0.0, 9.0, 6.0],
            vec![-3.0, 0.0, -9.0, -4.0],
        ]);
        let m_result = Matrix::new(vec![
            vec![-0.15385, -0.15385, -0.28205, -0.53846],
            vec![-0.07692, 0.12308, 0.02564, 0.03077],
            vec![0.35897, 0.35897, 0.43590, 0.92308],
            vec![-0.69231, -0.69231, -0.76923, -1.92308],
        ]);
        let new_m = m.inverse(0.0, -1.0);
        assert!(fmatrix_equals(m_result, new_m));

        let m = Matrix::new(vec![
            vec![9.0, 3.0, 0.0, 9.0],
            vec![-5.0, -2.0, -6.0, -3.0],
            vec![-4.0, 9.0, 6.0, 4.0],
            vec![-7.0, 6.0, 6.0, 2.0],
        ]);
        let m_result = Matrix::new(vec![
            vec![-0.04074, -0.07778, 0.14444, -0.22222],
            vec![-0.07778, 0.03333, 0.36667, -0.33333],
            vec![-0.02901, -0.14630, -0.10926, 0.12963],
            vec![0.17778, 0.06667, -0.26667, 0.33333],
        ]);
        let new_m = m.inverse(0.0, -1.0);
        assert!(fmatrix_equals(m_result, new_m));
    }

    #[test]
    fn test_matrix_mulidentity() {
        let m1 = Matrix::new(vec![
            vec![3.0, -9.0, 7.0, 3.0],
            vec![3.0, -8.0, 2.0, -9.0],
            vec![-4.0, 4.0, 4.0, 1.0],
            vec![-6.0, 5.0, -1.0, 1.0],
        ]);
        let m2 = Matrix::new(vec![
            vec![8.0, 2.0, 2.0, 2.0],
            vec![3.0, -1.0, 7.0, 0.0],
            vec![7.0, 0.0, 5.0, 4.0],
            vec![6.0, -2.0, 0.0, 5.0],
        ]);
        let m3 = &m1 * &m2;
        let m2_inverse = m2.inverse(0.0, -1.0);
        let new_m1 = &m3 * &m2_inverse;
        assert!(fmatrix_equals(m1, new_m1));
    }

    #[test]
    fn test_translate_point() {
        let m_translation = Matrix::<f64>::translation(5.0, -3.0, 2.0);
        let point = Tupple::point(-3.0, 4.0, 5.0);
        let new_point = point.translate(&m_translation);
        let result_point = Tupple::point(2.0, 1.0, 7.0);
        assert_eq!(new_point, result_point);

        let m_translation = Matrix::<f64>::translation(5.0, -3.0, 2.0);
        let m_inv = m_translation.inverse(0.0, -1.0);
        let point = Tupple::point(-3.0, 4.0, 5.0);
        let new_point = point.translate(&m_inv);
        let result_point = Tupple::point(-8.0, 7.0, 3.0);
        assert_eq!(new_point, result_point);

        let m_translation = Matrix::<f64>::translation(5.0, -3.0, 2.0);
        let point = Tupple::vector(-3.0, 4.0, 5.0);
        let new_point = point.translate(&m_translation);
        assert_eq!(new_point, point);
    }

    #[test]
    fn test_scaling_point() {
        let m_translation = Matrix::<f64>::scaling(2.0, 3.0, 4.0);
        let point = Tupple::point(-4.0, 6.0, 8.0);
        let new_point = point.translate(&m_translation);
        let result_point = Tupple::point(-8.0, 18.0, 32.0);
        assert_eq!(new_point, result_point);

        let m_translation = Matrix::<f64>::scaling(2.0, 3.0, 4.0);
        let vector = Tupple::vector(-4.0, 6.0, 8.0);
        let new_vector = vector.translate(&m_translation);
        let result_vector = Tupple::vector(-8.0, 18.0, 32.0);
        assert_eq!(new_vector, result_vector);

        let m_translation = Matrix::<f64>::scaling(2.0, 3.0, 4.0);
        let m_translation = m_translation.inverse(0.0, -1.0);
        let vector = Tupple::vector(-4.0, 6.0, 8.0);
        let new_vector = vector.translate(&m_translation);
        let result_vector = Tupple::vector(-2.0, 2.0, 2.0);
        assert_eq!(new_vector, result_vector);
    }

    #[test]
    fn test_reflection_point() {
        let m_translation = Matrix::<f64>::scaling(-1.0, 1.0, 1.0);
        let point = Tupple::point(2.0, 3.0, 4.0);
        let new_point = point.translate(&m_translation);
        let result_point = Tupple::point(-2.0, 3.0, 4.0);
        assert_eq!(new_point, result_point);
    }

    #[test]
    fn test_rotation_x() {
        let point = Tupple::point(0.0, 1.0, 0.0);
        let x_rotation_half_quarter = Matrix::<f64>::x_rotation(PI / 4.0);
        let x_rotation_full_quarter = Matrix::<f64>::x_rotation(PI / 2.0);
        let point2 = point.translate(&x_rotation_half_quarter);
        assert!(ftupple_equals(
            point2,
            Tupple::point(0.0, 2.0_f64.sqrt() / 2.0, 2.0_f64.sqrt() / 2.0),
        ));
        let point3 = point.translate(&x_rotation_full_quarter);
        assert!(ftupple_equals(point3, Tupple::point(0.0, 0.0, 1.0)));
    }

    #[test]
    fn test_inverse_rotation_x() {
        let point = Tupple::point(0.0, 1.0, 0.0);
        let x_rotation_half_quarter = Matrix::<f64>::x_rotation(PI / 4.0);
        let inv = x_rotation_half_quarter.inverse(0.0, -1.0);
        let point2 = point.translate(&inv);
        assert!(ftupple_equals(
            point2,
            Tupple::point(0.0, 2.0_f64.sqrt() / 2.0, -2.0_f64.sqrt() / 2.0),
        ));
    }

    #[test]
    fn test_rotation_y() {
        let point = Tupple::point(0.0, 0.0, 1.0);
        let y_rotation_half_quarter = Matrix::<f64>::y_rotation(PI / 4.0);
        let y_rotation_full_quarter = Matrix::<f64>::y_rotation(PI / 2.0);
        let point2 = point.translate(&y_rotation_half_quarter);
        assert!(ftupple_equals(
            point2,
            Tupple::point(2.0_f64.sqrt() / 2.0, 0.0, 2.0_f64.sqrt() / 2.0),
        ));
        let point3 = point.translate(&y_rotation_full_quarter);
        assert!(ftupple_equals(point3, Tupple::point(1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_rotation_z() {
        let point = Tupple::point(0.0, 1.0, 0.0);
        let z_rotation_half_quarter = Matrix::<f64>::z_rotation(PI / 4.0);
        let z_rotation_full_quarter = Matrix::<f64>::z_rotation(PI / 2.0);
        let point2 = point.translate(&z_rotation_half_quarter);
        assert!(ftupple_equals(
            point2,
            Tupple::point(-2.0_f64.sqrt() / 2.0, 2.0_f64.sqrt() / 2.0, 0.0),
        ));
        let point3 = point.translate(&z_rotation_full_quarter);
        assert!(ftupple_equals(point3, Tupple::point(-1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_shearing() {
        let point = Tupple::point(2.0, 3.0, 4.0);
        let shearing = Matrix::<f64>::shearing(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let point2 = point.translate(&shearing);
        assert_eq!(point2, Tupple::point(5.0, 3.0, 4.0));

        let point = Tupple::point(2.0, 3.0, 4.0);
        let shearing = Matrix::<f64>::shearing(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let point2 = point.translate(&shearing);
        assert_eq!(point2, Tupple::point(6.0, 3.0, 4.0));

        let point = Tupple::point(2.0, 3.0, 4.0);
        let shearing = Matrix::<f64>::shearing(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let point2 = point.translate(&shearing);
        assert_eq!(point2, Tupple::point(2.0, 5.0, 4.0));

        let point = Tupple::point(2.0, 3.0, 4.0);
        let shearing = Matrix::<f64>::shearing(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let point2 = point.translate(&shearing);
        assert_eq!(point2, Tupple::point(2.0, 7.0, 4.0));

        let point = Tupple::point(2.0, 3.0, 4.0);
        let shearing = Matrix::<f64>::shearing(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let point2 = point.translate(&shearing);
        assert_eq!(point2, Tupple::point(2.0, 3.0, 6.0));

        let point = Tupple::point(2.0, 3.0, 4.0);
        let shearing = Matrix::<f64>::shearing(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let point2 = point.translate(&shearing);
        assert_eq!(point2, Tupple::point(2.0, 3.0, 7.0));
    }

    #[test]
    fn test_chaining_transformations() {
        let point = point!(1, 0, 1);
        let x_rotation = rotation_x!(PI / 2.0);
        let scaling = scaling!(5, 5, 5);
        let translation = translation!(10, 5, 7);
        let new_point = point
            .translate(&x_rotation)
            .translate(&scaling)
            .translate(&translation);
        assert!(ftupple_equals(new_point, point!(15, 0, 7)));
    }

    #[test]
    fn test_rays() {
        let ray = Ray::new(point!(2, 3, 4), vector!(1, 0, 0));
        assert_eq!(ray.position(0.0), point!(2, 3, 4));
        assert_eq!(ray.position(1.0), point!(3, 3, 4));
        assert_eq!(ray.position(-1.0), point!(1, 3, 4));
        assert_eq!(ray.position(2.5), point!(4.5, 3, 4));
    }

    #[test]
    fn test_intersect() {
        let ray = ray!(point!(0, 1, -5), vector!(0, 0, 1));
        let sphere = Sphere::new();
        let xs = sphere.intersect(ray);
        assert!(xs[0] == 5.0);
        assert!(xs[1] == 5.0);

        let ray = ray!(point!(0, 2, -5), vector!(0, 0, 1));
        let sphere = Sphere::new();
        let xs = sphere.intersect(ray);
        assert!(xs.len() == 0);

        let ray = ray!(point!(0, 0, 0), vector!(0, 0, 1));
        let sphere = Sphere::new();
        let xs = sphere.intersect(ray);
        assert!(xs[0] == -1.0);
        assert!(xs[1] == 1.0);

        let ray = ray!(point!(0, 0, 5), vector!(0, 0, 1));
        let sphere = Sphere::new();
        let xs = sphere.intersect(ray);
        assert!(xs[0] == -6.0);
        assert!(xs[1] == -4.0);
    }
}
