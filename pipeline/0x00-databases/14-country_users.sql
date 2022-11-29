-- Creates a table "users" with 3 attributes: id (INT, NEVER NULL), email (VARCHAR 255, NEVER NULL & UNIQUE), name (VARCHAR 255)
CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    country ENUM('US', 'CO', 'TN') NOT NULL,
    PRIMARY KEY (id)
)