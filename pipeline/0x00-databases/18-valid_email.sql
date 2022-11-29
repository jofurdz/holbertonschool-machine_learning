-- Creates a trigger that resets the attribute valid_email only when email has been changed
DELIMITER //
CREATE TRIGGER change_email
    BEFORE UPDATE
    ON users
    FOR EACH ROW
        IF STRCMP(old.email, new.email) <> 0 THEN
            SET new.valid_email = 0
        END IF;
DELIMITER ;