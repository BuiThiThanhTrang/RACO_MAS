from tools.base.register import global_tool_registry
from tools.base.base_tool import Tool
from abc import abstractmethod
import signal
from functools import wraps

def timeout_handler(signum, frame):
    raise TimeoutError("Request timed out")

def timeout(seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set the signal handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

class Web_Search(Tool):
    def __init__(self):
        super().__init__("web_search", "Search the web for a given query", self.execute)
        self.broswer = None
        self.downloads_folder = ""

    def _get_browser(self):
        if self.broswer is None:
            try:
                from tools.utils.broswer import SimpleTextBrowser
            except ImportError as e:
                raise ImportError(
                    "Browser search dependencies are not installed. "
                    "Run: pip install -r requirements.txt"
                ) from e
            self.broswer = SimpleTextBrowser()
        self.broswer.downloads_folder = self.downloads_folder
        return self.broswer
    
    def execute(self,*args,**kwargs):
        try: 
            query = kwargs.get("query", "")
            self.downloads_folder = kwargs.get("work_path", "")
            flag, ans = self.search(query)
        except AttributeError:
            return False, "No results found for query {}".format(query)
        except TimeoutError:
            return False, "Timeout"
        except Exception as e:
            return False, "No results found for query {}".format(query)

        if (ans is None) or (len(ans) == 0):
            # raise ValueError(f"No results found for query {query}.")
            return False, "No results found for query {}".format(query)
        
        return flag, ans
    
    @abstractmethod
    def search(self, query):
        pass

import requests

@global_tool_registry("search_arxiv")
class arXiv_SearchEngine(Web_Search):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def search(self, query):
        try:
            import arxiv
        except ImportError:
            return False, "The 'arxiv' package is not installed. Run: pip install -r requirements.txt"

        # Custom timeout
        timeout = 10  # Timeout in seconds

        # Create a custom session with a timeout
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(timeout=timeout)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        try:
            # Perform the search with custom session
            search = arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance,
                session=session  # Use the session with timeout
            )
            
            results = []
            for result in search.results():
                result_info = {
                    "title": result.title,
                    "authors": ", ".join(author.name for author in result.authors),
                    "summary": result.summary,
                    "pdf_url": result.pdf_url
                }
                results.append(result_info)
            
            if len(results) == 0:
                results = "Page not exists in arXiv, try different search tools like Bing search."
                return False, results

            return True, str(results)

        except requests.exceptions.Timeout:
            return False, "Request timed out. Please try again later."

        except Exception as e:
            return False, f"An error occurred: {e}"
    

@global_tool_registry("search_bing")
class Bing_SearchEngine(Web_Search):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def search(self, query):
        try:
            broswer = self._get_browser()
        except ImportError as e:
            return False, str(e)

        broswer.set_address("bing:"+query)
        if broswer.page_content != None and len(broswer.page_content) != 0:
            return True, broswer.page_content
        else:
            return False, "page not exists in bing, try different search tools"


@global_tool_registry("access_website")
class Website_SearchEngine(Web_Search):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def search(self, url):
        try:
            broswer = self._get_browser()
        except ImportError as e:
            return False, str(e)

        broswer.set_address(url)
        if broswer.page_content != None and len(broswer.page_content) != 0:
            if "Failed to fetch" in broswer.page_content:
                return False, broswer.page_content
            else:
                return True, broswer.page_content
        else:
            return False, "Can not Access this website: {}".format(url)
