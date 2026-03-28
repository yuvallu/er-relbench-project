create table Pub (k text, p text);
create table Field (k text, i text, p text, v text);
copy Pub from '/Users/suciu/PROJECTS-UNISON/COURSES/544/2012/SVN/cse544_12sp/assignments/hw1/solution/pubFile.txt';
copy Field from '/Users/suciu/PROJECTS-UNISON/COURSES/544/2012/SVN/cse544_12sp/assignments/hw1/solution/fieldFile.txt';
