CREATE TABLE table_a (
    foo,
    bar INTEGER,
    boo INTEGER
);

CREATE TABLE table_b (
    foo INTEGER,
    bar,
    boo INTEGER
);

CREATE TABLE table_c (
    foo INTEGER,
    bar INTEGER,
    boo
);

INSERT INTO table_a VALUES
  (1, 2, 3),
  ('hello', 3, 4),
  (3, 4, 5);

INSERT INTO table_b VALUES
  (1, 2, 3),
  (2, 'hello', 4),
  (3, 4, 5);

INSERT INTO table_c VALUES
  (1, 2, 3),
  (2, 3, 'hello'),
  (3, 4, 5);