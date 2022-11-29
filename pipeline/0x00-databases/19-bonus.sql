-- Creates a stored procedure AddBonus that adds a new correction for a student
DELIMITER //
    CREATE PROCEDURE AddBonus (user_id INT, project_name VARCHAR(255), score FLOAT)
    BEGIN
        IF NOT EXISTS (SELECT name FROM projects WHERE name=project_name) THEN
            INSERT INTO projects(name) VALUE (project_name);
        END IF;

        SET @project_id = (SELECT id FROM projects WHERE name = project_name);

        INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, @project_id, score);
    END //
DELIMITER ;