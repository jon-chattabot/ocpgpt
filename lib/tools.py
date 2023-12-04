from pytz import timezone
from dateutil import parser
from re import compile
from datetime import datetime
from calendar import day_name
from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class TodaysDateTool(BaseTool):
    name = "todays_date_tool"
    description="useful for when you need to get current date"

    @staticmethod
    def today():
        eastern = timezone('US/Eastern')
        return datetime.now(eastern).strftime('%m/%d/%y')

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        eastern = timezone('US/Eastern')
        return TodaysDateTool.today()

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async")

class DayOfTheWeekTool(BaseTool):
    name = "day_of_the_week"
    description="useful for when you need the day of the week for a relative date like today or tomorrow"

    @staticmethod
    def today():
        eastern = timezone('US/Eastern')
        return day_name[datetime.now(eastern).weekday()]

    def day_of_week(self, date:str) -> str:
        if date == 'today':
            return DayOfTheWeekTool.today()

        date_pattern = compile(r'^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{2}$')
        if date_pattern.match(date):
            try:
                the_date = parser.parse(date)
                return day_name[the_date.weekday()]
            except:
                return 'invalid date, unable to parse'
        else:
            return 'invalid date format, please use format: mm/dd/yy'

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self.day_of_week(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async")

# def handle_error(error: ToolException) -> str:
    # return (
        # "The following errors occurred during tool execution:"
        # + error.args[0]
        # + "Please try another tool."
    # )

