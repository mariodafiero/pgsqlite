/*
This schema is intended to exercise all variations of sqlite quoting across
table and column identifiers including within foreign keys and indexes.
*/

-- lowercase identifiers, no quoting
CREATE TABLE parent_a (
    id INTEGER PRIMARY KEY,
    foo TEXT,
    bar TEXT,
    baz TEXT
);

-- uppercase identifiers, no quoting
CREATE TABLE PARENT_B (
    ID INTEGER PRIMARY KEY,
    FOO TEXT,
    BAR TEXT,
    BAZ TEXT
);

-- uppercase identifiers, double-quoting
CREATE TABLE "PARENT_C" (
    "ID" INTEGER PRIMARY KEY,
    "FOO" TEXT,
    "BAR" TEXT,
    "BAZ" TEXT
);

-- uppercase identifiers, bracket-quoting
CREATE TABLE [PARENT_D] (
    [ID] INTEGER PRIMARY KEY,
    [FOO] TEXT,
    [BAR] TEXT,
    [BAZ] TEXT
);

-- uppercase identifiers, backtick-quoting
CREATE TABLE `PARENT_E` (
    `ID` INTEGER PRIMARY KEY,
    `FOO` TEXT,
    `BAR` TEXT,
    `BAZ` TEXT
);

-- non-ascii, quoted
CREATE TABLE "parent_f@/" (
    "id@/" INTEGER PRIMARY KEY,
    foo TEXT,
    bar TEXT,
    baz TEXT
);

-- child table referencing all parent tables
CREATE TABLE child (
    child_id INTEGER PRIMARY KEY,
    child_unique TEXT UNIQUE, -- test our ability to handle unique constraints
    a_id INTEGER REFERENCES parent_a (id),
    b_id INTEGER REFERENCES PARENT_B (ID),
    c_id INTEGER REFERENCES "PARENT_C" ("ID"),
    d_id INTEGER REFERENCES [PARENT_D] ([ID]),
    e_id INTEGER REFERENCES `PARENT_E` (`ID`),
    f_id INTEGER REFERENCES "parent_f@/" ("id@/")
);

-- also test single and multi-column indexes with all variations
CREATE INDEX parent_a_single_col ON parent_a (foo);
CREATE INDEX parent_a_multi_col ON parent_a (bar, baz);

CREATE INDEX parent_b_single_col ON PARENT_B (FOO);
CREATE INDEX parent_b_multi_col ON PARENT_B (BAR, BAZ);

CREATE INDEX parent_c_single_col ON "PARENT_C" ("FOO");
CREATE INDEX parent_c_multi_col ON "PARENT_C" ("BAR", "BAZ");

CREATE INDEX parent_d_single_col ON [PARENT_D] ([FOO]);
CREATE INDEX parent_d_multi_col ON [PARENT_D] ([BAR], [BAZ]);

CREATE INDEX parent_e_single_col ON `PARENT_E` (`FOO`);
CREATE INDEX parent_e_multi_col ON `PARENT_E` (`BAR`, `BAZ`);

CREATE INDEX parent_f_single_col ON "parent_f@/" ("id@/");
CREATE INDEX parent_f_multi_col ON "parent_f@/" ("id@/", foo);

/* 
In addition to double-quotes, brackets, and backticks, sqlite also "unofficially"
accepts single-quotes quoting of identifiers when the context disambiguates from
a string literal. However, the sqlite maintainers note that this feature should
not be relied upon and could change without notice. Replicating that tolerance
in the sqlite dialect of the sqlglot parser doesn't seem worth it. For now,
pgsqlite will not attempt to handle that condition beyond potentially adding
informative errors when detected.
*/

INSERT INTO parent_a VALUES
  (1, "apple", "orange", "banana"),
  (2, "orange", "apple", "banana");

INSERT INTO PARENT_B VALUES
  (1, "apple", "orange", "banana"),
  (2, "orange", "apple", "banana");

INSERT INTO "PARENT_C" VALUES
  (1, "apple", "orange", "banana"),
  (2, "orange", "apple", "banana");

INSERT INTO [PARENT_D] VALUES
  (1, "apple", "orange", "banana"),
  (2, "orange", "apple", "banana");

INSERT INTO `PARENT_E` VALUES
  (1, "apple", "orange", "banana"),
  (2, "orange", "apple", "banana");

INSERT INTO "parent_f@/" VALUES
  (1, "apple", "orange", "banana"),
  (2, "orange", "apple", "banana");

INSERT INTO child VALUES
  (1, 'a', 1, 1, 1, 1, 1, 1),
  (2, 'b', 2, 2, 2, 2, 2, 2);