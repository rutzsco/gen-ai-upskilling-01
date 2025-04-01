
class FileService:
    """
    FileService is a class that provides functionality to manage and read files 
    using a mapping of file names to their full paths.
    Methods:
        __init__():
            Initializes the FileService instance with an empty file map.
        add_file(file_name: str, file_path: str):
            Adds a file to the file map by associating the given file name with its full path.
            Args:
                file_name (str): The name of the file to add.
                file_path (str): The full path of the file to add.
        read_file(file_name: str) -> str:
            Reads the content of a file based on its name in the file map.
            Args:
                file_name (str): The name of the file to read.
            Returns:
                str: The content of the file if it exists and is readable.
                     If the file is not found in the file map or at the specified path, 
                     an appropriate error message is returned.
    """
    def __init__(self):
        # Initialize a dictionary to map file names to their full paths
        self.file_map = {}
        self.add_file('RAGSystemPrompt.txt', './app/prompt/RAGSystemPrompt.txt') 
        self.add_file('RAGSearchSystemPrompt.txt', './app/prompt/RAGSearchSystemPrompt.txt') 
        self.add_file('RAGAgentSystemPrompt.txt', './app/prompt/RAGAgentSystemPrompt.txt') 
    def add_file(self, file_name, file_path):
        self.file_map[file_name] = file_path

    def read_file(self, file_name):
        if file_name in self.file_map:
            file_path = self.file_map[file_name]
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                return content
            except FileNotFoundError:
                raise RuntimeError( f"File '{file_name}' not found at path '{file_path}'.")
        else:
            raise RuntimeError(f"File '{file_name}' not found in the file map.")
