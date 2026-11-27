using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace Eval;

public sealed record Todo(int Id, string Title, bool Done);

public static class TodoApi
{
    public static void MapTodoEndpoints(IEndpointRouteBuilder app)
    {
        throw new NotImplementedException("Endpointy nie su implementovane.");
    }
}
