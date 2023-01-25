/*
sqlite accepts "typeless" columns and stores the associated data as text.
Therefore, we should import the following tables to Postgres w/ typeless
columns as Postgres TEXT columns.
*/

CREATE TABLE table_a (
    foo,
    -- Testing an interleaving comment
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

CREATE TABLE table_d (
    foo INTEGER,
    bar,
    boo INTEGER,
    baz
);

INSERT INTO table_a VALUES
  (1, 2, 3),
  ('hello', 2, 3),
  (1, 2, 3);

INSERT INTO table_b VALUES
  (1, 2, 3),
  (1, 'hello', 3),
  (1, 2, 3);

INSERT INTO table_c VALUES
  (1, 2, 3),
  (1, 2, 'hello'),
  (1, 2, 3);

INSERT INTO table_d VALUES
  (1, 2, 3, 4),
  (1, 'hello', 3, 'world'),
  (1, 2, 3, 4),
  (1, 'hello', 3, 'world');