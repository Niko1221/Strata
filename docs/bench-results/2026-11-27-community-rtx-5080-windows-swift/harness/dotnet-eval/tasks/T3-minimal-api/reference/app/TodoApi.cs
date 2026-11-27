using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace Eval;

public sealed record Todo(int Id, string Title, bool Done);

public static class TodoApi
{
    public sealed record CreateTodo(string? Title);

    private static readonly List<Todo> Items = new();
    private static readonly object Gate = new();
    private static int _next = 1;

    public static void MapTodoEndpoints(IEndpointRouteBuilder app)
    {
        app.MapGet("/todos", IResult () =>
        {
            lock (Gate)
            {
                return Results.Ok(Items.ToArray());
            }
        });

        app.MapGet("/todos/{id:int}", IResult (int id) =>
        {
            lock (Gate)
            {
                var todo = Items.FirstOrDefault(x => x.Id == id);
                return todo is null ? Results.NotFound() : Results.Ok(todo);
            }
        });

        app.MapPost("/todos", IResult (CreateTodo? body) =>
        {
            if (body is null || string.IsNullOrWhiteSpace(body.Title))
            {
                return Results.BadRequest();
            }

            Todo created;
            lock (Gate)
            {
                created = new Todo(_next++, body.Title.Trim(), false);
                Items.Add(created);
            }

            return Results.Created($"/todos/{created.Id}", created);
        });

        app.MapPut("/todos/{id:int}/done", IResult (int id) =>
        {
            lock (Gate)
            {
                var index = Items.FindIndex(x => x.Id == id);
                if (index < 0)
                {
                    return Results.NotFound();
                }

                Items[index] = Items[index] with { Done = true };
                return Results.NoContent();
            }
        });

        app.MapDelete("/todos/{id:int}", IResult (int id) =>
        {
            lock (Gate)
            {
                var index = Items.FindIndex(x => x.Id == id);
                if (index < 0)
                {
                    return Results.NotFound();
                }

                Items.RemoveAt(index);
                return Results.NoContent();
            }
        });
    }
}
