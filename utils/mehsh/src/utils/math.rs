#[must_use]
#[inline]
pub fn calculate_average_f32<T>(list: impl Iterator<Item = T>) -> T
where
    T: Default + std::ops::Add<Output = T> + std::ops::Div<f32, Output = T> + std::iter::Sum<T>,
{
    let (sum, count) = list.fold((T::default(), 0.0), |(sum, count), elem| (sum + elem, count + 1.0));
    sum / count
}

#[must_use]
#[inline]
pub fn calculate_average_f64<T>(list: impl Iterator<Item = T>) -> T
where
    T: Default + std::ops::Add<Output = T> + std::ops::Div<f64, Output = T> + std::iter::Sum<T>,
{
    let (sum, count) = list.fold((T::default(), 0.0), |(sum, count), elem| (sum + elem, count + 1.0));
    sum / count
}
