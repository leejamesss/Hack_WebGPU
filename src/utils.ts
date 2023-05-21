

export function createFromSize(sizes: number[], default_value: number = 0): any {
  if (sizes.length === 0) {
    return default_value;
  }
  const size = sizes[0];
  const rest = sizes.slice(1);
  const arr = new Array(size).fill(default_value);
  for (let i = 0; i < size; i++) {
    arr[i] = createFromSize(rest,default_value);
  }
  return arr;
}

export function createFromSizeUniform(sizes: number[], min: number = -1,max: number=1): any {
  if (sizes.length === 0) {
    return min+ (max-min)*Math.random();
  }
  const size = sizes[0];
  const rest = sizes.slice(1);
  const arr = new Array(size).fill(undefined);
  for (let i = 0; i < size; i++) {
    arr[i] = createFromSizeUniform(rest,min,max);
  }
  return arr;
}

export function range(size: Number, startAt = 0) {
    return [...Array(size).keys()].map(i => i + startAt);
}
export function* product<T> (...pools: T[][]): IterableIterator<T[]> {
// function* product (...pools) {
  let i = 0;
  const indexes = new Array(pools.length).fill(0)
  const result = indexes.map((x, i) => pools[i][x]);
  indexes[0] = -1;
  while (i < indexes.length) {
    if (indexes[i] < pools[i].length - 1) {
      indexes[i]++;
      result[i] = pools[i][indexes[i]];
      i = 0;

      // NB: can remove `.slice()` for a performance improvement if you don't mind mutating the same result object
      yield result.slice();
    } else {
      indexes[i] = 0;
      result[i] = pools[i][0];
      i++;
    }
  }
}