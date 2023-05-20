
export function createFromSize(sizes: number[], default_value: number = 0): any {
  if (sizes.length === 0) {
    return 0;
  }
  const size = sizes[0];
  const rest = sizes.slice(1);
  const arr = new Array(size).fill(default_value);
  for (let i = 0; i < size; i++) {
    arr[i] = createFromSize(rest);
  }
  return arr;
}