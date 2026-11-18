-- Runs once, when the Postgres volume is first created.
-- rag_test is for the Postgres integration tests, which truncate tables;
-- they never touch the main database "rag".
CREATE DATABASE rag_test;
